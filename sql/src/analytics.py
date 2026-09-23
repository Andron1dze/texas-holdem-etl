"""
analytics.py — аналитический этап проекта «Texas Hold'em ETL & Analytics Pipeline».

Что делает скрипт:
    1. Подключается к `data/poker_analytics.db` и устанавливает витрины
       из `sql/analytics_queries.sql` (v_player_stats, v_position_winrate).
    2. Забирает витрины в pandas и досчитывает то, что удобнее в Python:
       стандартную ошибку винрейта (95% доверительный интервал).
    3. Сверяет стиль игроков, восстановленный по статистике, со скрытым
       архетипом из генератора (`data/raw/player_profiles.csv`) — проверка того,
       что метрики действительно «видят» стиль игры.
    4. Строит графики в `reports/figures/`:
         * vpip_pfr_scatter.png  — карта стилей: VPIP × PFR, цвет = bb/100;
         * winrate_by_position.png — винрейт пула по позициям с 95% ДИ.
    5. Сохраняет витрину игроков в `reports/player_stats.csv`.

Запуск из корня репозитория (после `python src/data_loader.py`):
    python src/analytics.py
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

logger = logging.getLogger("analytics")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# =============================================================================
#  ОФОРМЛЕНИЕ ГРАФИКОВ
# =============================================================================
# Палитра подобрана под читаемость: нейтральные «чернила» для текста и осей,
# расходящаяся шкала красный ↔ серый ↔ синий для знака винрейта
# (серый центр = «ноль», а не отдельная категория).

INK_PRIMARY: Final[str] = "#0b0b0b"
INK_SECONDARY: Final[str] = "#52514e"
INK_MUTED: Final[str] = "#898781"
GRID_COLOR: Final[str] = "#e1e0d9"
BASELINE_COLOR: Final[str] = "#c3c2b7"
SURFACE: Final[str] = "#fcfcfb"

LOSS_COLOR: Final[str] = "#d03b3b"       # отрицательный винрейт
WIN_COLOR: Final[str] = "#2a78d6"        # положительный винрейт
NEUTRAL_MID: Final[str] = "#e4e2dc"      # около нуля

WINRATE_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    "winrate", ["#9e2a2a", LOSS_COLOR, NEUTRAL_MID, WIN_COLOR, "#1c5cab"]
)

# Маркер кодирует стиль, чтобы идентичность не держалась только на цвете
STYLE_MARKERS: Final[dict[str, str]] = {
    "Nit": "s", "TAG": "o", "LAG": "D", "Loose-Passive": "^",
}
STYLE_ORDER: Final[tuple[str, ...]] = ("Nit", "TAG", "LAG", "Loose-Passive")

# Архетип генератора → стиль, который должна распознать аналитика
ARCHETYPE_TO_STYLE: Final[dict[str, str]] = {
    "Nit": "Nit", "TAG": "TAG", "LAG": "LAG", "Fish": "Loose-Passive",
}

POSITION_LABELS: Final[dict[str, str]] = {
    "UTG": "UTG", "HJ": "HJ (MP)", "CO": "CO", "BTN": "BTN", "SB": "SB", "BB": "BB",
}

Z_95: Final[float] = 1.96          # квантиль нормального распределения для 95% ДИ
N_LABELED_EXTREMES: Final[int] = 3  # сколько лидеров и аутсайдеров подписать на scatter
LABEL_MIN_DISTANCE: Final[float] = 3.0  # в процентных пунктах VPIP/PFR


@dataclass(frozen=True, slots=True)
class AnalyticsConfig:
    """Пути ввода-вывода аналитического этапа."""

    db_path: Path = PROJECT_ROOT / "data" / "poker_analytics.db"
    sql_path: Path = PROJECT_ROOT / "sql" / "analytics_queries.sql"
    profiles_path: Path = PROJECT_ROOT / "data" / "raw" / "player_profiles.csv"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


def apply_chart_style() -> None:
    """Единый стиль для всех графиков: тихие оси и сетка, акцент — на данных."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE_COLOR,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "font.size": 10,
    })


# =============================================================================
#  ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ SQLITE
# =============================================================================

def open_database(db_path: Path) -> sqlite3.Connection:
    """Открывает БД только на чтение данных (представления создаются отдельно)."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found — run `python src/data_loader.py` first")
    return sqlite3.connect(db_path)


def install_views(conn: sqlite3.Connection, sql_path: Path) -> None:
    """Выполняет analytics_queries.sql: пересоздаёт витрины (идемпотентно).

    Итоговые SELECT в конце файла executescript выполняет и отбрасывает —
    они предназначены для ручного запуска в SQL-клиенте (DBeaver, sqlite3).
    """
    conn.executescript(sql_path.read_text(encoding="utf-8"))
    logger.info("Views installed from %s", sql_path.name)


def fetch_player_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """Витрина профилей игроков, отсортированная по винрейту."""
    return pd.read_sql_query("SELECT * FROM v_player_stats ORDER BY winrate_rank", conn)


def fetch_position_winrate(conn: sqlite3.Connection) -> pd.DataFrame:
    """Витрина винрейта по позициям + 95% доверительный интервал bb/100."""
    positions = pd.read_sql_query("SELECT * FROM v_position_winrate ORDER BY preflop_order", conn)
    return positions.assign(ci95_bb_per_100=winrate_ci95(
        positions["avg_net_bb"].to_numpy(),
        positions["avg_sq_net_bb"].to_numpy(),
        positions["hands"].to_numpy(),
    ))


def winrate_ci95(
    mean: npt.NDArray[np.float64],
    mean_of_squares: npt.NDArray[np.float64],
    n: npt.NDArray[np.int_],
) -> npt.NDArray[np.float64]:
    """Полуширина 95% ДИ для bb/100 по моментам E[x] и E[x²] (векторно).

    Дисперсия выборки: Var = E[x²] − E[x]², с поправкой Бесселя n/(n−1).
    """
    variance = np.clip(mean_of_squares - mean**2, 0.0, None) * n / np.maximum(n - 1, 1)
    return Z_95 * 100.0 * np.sqrt(variance / n)


# =============================================================================
#  ВАЛИДАЦИЯ АНАЛИТИКИ ПО GROUND TRUTH
# =============================================================================

def compare_with_ground_truth(stats: pd.DataFrame, profiles_path: Path) -> pd.DataFrame:
    """Таблица сопряжённости «скрытый архетип × стиль по статистике».

    Если файла профилей нет (например, данные реальные) — возвращает пустой DataFrame.
    """
    if not profiles_path.exists():
        logger.warning("Ground truth not found (%s) — skipping comparison", profiles_path)
        return pd.DataFrame()

    profiles = pd.read_csv(profiles_path, usecols=["name", "archetype"])
    merged = stats.merge(profiles, left_on="player_name", right_on="name", how="inner")
    expected = merged["archetype"].map(ARCHETYPE_TO_STYLE)
    accuracy = float((expected == merged["player_style"]).mean())

    confusion = pd.crosstab(merged["archetype"], merged["player_style"]).reindex(
        columns=list(STYLE_ORDER), fill_value=0
    )
    logger.info("Style recognition accuracy vs ground truth: %.0f%% (%d players)\n%s",
                100 * accuracy, len(merged), confusion.to_string())
    return confusion


# =============================================================================
#  ВИЗУАЛИЗАЦИЯ
# =============================================================================

def plot_vpip_pfr_scatter(stats: pd.DataFrame) -> Figure:
    """Карта стилей игроков: X — VPIP, Y — PFR, цвет — bb/100, маркер — стиль."""
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")

    limit = float(np.ceil(max(stats["vpip_pct"].max(), stats["pfr_pct"].max()) / 5) * 5 + 5)
    _draw_pfr_ceiling(ax, limit)

    # Симметричная шкала цвета вокруг нуля: одинаковая «сила» красного и синего
    span = float(stats["bb_per_100"].abs().max())
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    for style in STYLE_ORDER:
        group = stats[stats["player_style"] == style]
        if group.empty:
            continue
        ax.scatter(
            group["vpip_pct"], group["pfr_pct"],
            c=group["bb_per_100"], cmap=WINRATE_CMAP, norm=norm,
            marker=STYLE_MARKERS[style], s=110,
            edgecolors=INK_SECONDARY, linewidths=0.8, zorder=3,
        )

    _label_extremes(ax, stats)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=WINRATE_CMAP), ax=ax, shrink=0.8, pad=0.02
    )
    colorbar.set_label("Винрейт, bb/100", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)

    ax.legend(handles=_style_legend_handles(stats), title="Стиль по статистике",
              title_fontsize=9.5, loc="upper left", alignment="left")
    ax.set(xlim=(0, limit), ylim=(0, limit * 0.8),
           xlabel="VPIP, % — как часто игрок добровольно входит в банк",
           ylabel="PFR, % — как часто рейзит префлоп")
    ax.set_title("Карта стилей игроков: VPIP × PFR")
    return fig


def _draw_pfr_ceiling(ax: Axes, limit: float) -> None:
    """Диагональ PFR = VPIP: физический потолок (рейз — частный случай VPIP)."""
    ax.plot([0, limit], [0, limit], color=BASELINE_COLOR, linestyle="--", linewidth=1, zorder=1)
    ax.annotate("PFR = VPIP (каждый вход — рейзом)", xy=(limit * 0.52, limit * 0.52),
                xytext=(6, -14), textcoords="offset points",
                rotation=36, color=INK_MUTED, fontsize=8.5)


def _label_extremes(ax: Axes, stats: pd.DataFrame) -> None:
    """Подписывает только лидеров и аутсайдеров по винрейту, не каждую точку.

    Простейшее разведение подписей: если рядом уже есть подпись
    (ближе LABEL_MIN_DISTANCE в единицах осей), новая уходит под точку.
    """
    ranked = stats.sort_values("bb_per_100")
    extremes = pd.concat([ranked.head(N_LABELED_EXTREMES), ranked.tail(N_LABELED_EXTREMES)])
    placed: list[npt.NDArray[np.float64]] = []
    for row in extremes.itertuples(index=False):
        point = np.array([row.vpip_pct, row.pfr_pct])
        crowded = any(np.linalg.norm(point - other) < LABEL_MIN_DISTANCE for other in placed)
        ax.annotate(f"{row.player_name}  {row.bb_per_100:+.0f}",
                    xy=tuple(point), xytext=(8, -14 if crowded else 5), textcoords="offset points",
                    fontsize=8.5, color=INK_SECONDARY, zorder=4)
        placed.append(point)


def _style_legend_handles(stats: pd.DataFrame) -> list[Line2D]:
    """Легенда по форме маркера с числом игроков каждого стиля."""
    counts = stats["player_style"].value_counts()
    return [
        Line2D([], [], marker=STYLE_MARKERS[style], linestyle="none", markersize=8,
               markerfacecolor=NEUTRAL_MID, markeredgecolor=INK_SECONDARY,
               label=f"{style} ({counts.get(style, 0)})")
        for style in STYLE_ORDER if counts.get(style, 0) > 0
    ]


def plot_position_winrate(positions: pd.DataFrame) -> Figure:
    """Средний винрейт пула по позициям (порядок хода префлоп) с 95% ДИ."""
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")

    x = np.arange(len(positions))
    winrate = positions["bb_per_100"].to_numpy()
    ci = positions["ci95_bb_per_100"].to_numpy()
    colors = np.where(winrate >= 0, WIN_COLOR, LOSS_COLOR)

    ax.bar(x, winrate, width=0.62, color=colors, zorder=2)
    ax.errorbar(x, winrate, yerr=ci, fmt="none", ecolor=INK_SECONDARY,
                elinewidth=1, capsize=4, zorder=3)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1, zorder=2)

    # Значение — в подписи оси: не спорит с «усами» ДИ и читается сразу с позицией
    tick_labels = [
        f"{POSITION_LABELS.get(code, code)}\n{value:+.1f}"
        for code, value in zip(positions["position_code"], winrate)
    ]
    ax.set_xticks(x, tick_labels)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.margins(y=0.2)
    ax.set_ylabel("Винрейт, bb/100")
    ax.set_title("Винрейт по позициям  ·  усы — 95% доверительный интервал")
    sample = f"{int(positions['hands'].iloc[0]):,}".replace(",", "\u202f")
    ax.text(0, -0.16, f"Выборка: {sample} раздач на позицию. "
            "Блайнды теряют из-за обязательных ставок, баттон выигрывает за счёт позиции.",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED)
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info("Figure saved: %s", path.relative_to(PROJECT_ROOT))


# =============================================================================
#  ОРКЕСТРАЦИЯ
# =============================================================================

def run_analytics(config: AnalyticsConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Полный аналитический прогон. Возвращает витрины игроков и позиций."""
    apply_chart_style()
    with closing(open_database(config.db_path)) as conn:
        install_views(conn, config.sql_path)
        stats = fetch_player_stats(conn)
        positions = fetch_position_winrate(conn)

    logger.info("Top players by bb/100:\n%s", stats.head(5)[
        ["player_name", "hands", "vpip_pct", "pfr_pct", "three_bet_pct",
         "aggression_factor", "bb_per_100", "player_style"]
    ].to_string(index=False))
    compare_with_ground_truth(stats, config.profiles_path)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(config.reports_dir / "player_stats.csv", index=False)
    save_figure(plot_vpip_pfr_scatter(stats), config.figures_dir / "vpip_pfr_scatter.png")
    save_figure(plot_position_winrate(positions), config.figures_dir / "winrate_by_position.png")
    return stats, positions


def parse_args() -> AnalyticsConfig:
    defaults = AnalyticsConfig()
    parser = argparse.ArgumentParser(description="Build poker analytics marts and charts.")
    parser.add_argument("--db", type=Path, default=defaults.db_path, help="SQLite file path")
    parser.add_argument("--reports", type=Path, default=defaults.reports_dir, help="output directory")
    args = parser.parse_args()
    return AnalyticsConfig(db_path=args.db, reports_dir=args.reports)


def main() -> None:
    # Рендер без GUI только при запуске скриптом: при импорте из ноутбука
    # бэкенд не трогаем, чтобы графики Jupyter отображались inline.
    plt.switch_backend("Agg")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    run_analytics(parse_args())


if __name__ == "__main__":
    main()
