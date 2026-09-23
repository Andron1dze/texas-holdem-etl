"""
test_data_quality.py — автотесты качества данных для poker_analytics.db.

Проверяется не код, а САМИ ДАННЫЕ: бизнес-инварианты покера, ссылочная
целостность и математическая адекватность метрик витрины. Такие тесты ловят
ошибки, которые не видны в юнит-тестах функций: неверную агрегацию в SQL,
рассинхронизацию схемы и витрины, «битые» строки после ручной правки БД.

Уровни проверок:
    * инварианты раздачи  — zero-sum, банк = сумма ставок, 6 мест за столом;
    * витрина v_player_stats — метрики в допустимых диапазонах и согласованы
      между собой (PFR ≤ VPIP);
    * целостность связей  — действия принадлежат игрокам, сидевшим в раздаче;
    * контракт схемы      — БД физически отвергает некорректную запись.

Запуск (из корня репозитория, после `python src/data_loader.py`):
    pytest tests/ -v
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DB_PATH: Final[Path] = PROJECT_ROOT / "data" / "poker_analytics.db"
ANALYTICS_SQL_PATH: Final[Path] = PROJECT_ROOT / "sql" / "analytics_queries.sql"

SEATS_PER_HAND: Final[int] = 6
PERCENTAGE_BOUNDS: Final[tuple[float, float]] = (0.0, 100.0)
MAX_REPORTED_ROWS: Final[int] = 5      # сколько «плохих» строк показывать в сообщении об ошибке


# =============================================================================
#  ФИКСТУРЫ
# =============================================================================

@pytest.fixture(scope="session")
def connection() -> Iterator[sqlite3.Connection]:
    """Одно подключение к БД на весь прогон тестов (с включёнными внешними ключами)."""
    if not DB_PATH.exists():
        pytest.fail(f"База {DB_PATH} не найдена — сначала выполните `python src/data_loader.py`")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session")
def player_stats(connection: sqlite3.Connection) -> pd.DataFrame:
    """Витрина игроков. Представления пересоздаются здесь же — тесты самодостаточны."""
    connection.executescript(ANALYTICS_SQL_PATH.read_text(encoding="utf-8"))
    stats = pd.read_sql_query("SELECT * FROM v_player_stats", connection)
    assert not stats.empty, "Витрина v_player_stats пуста"
    return stats


def fetch(connection: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Читает результат запроса в DataFrame (запросы ниже возвращают только нарушения)."""
    return pd.read_sql_query(sql, connection)


def describe(violations: pd.DataFrame) -> str:
    """Первые несколько нарушений — в текст сообщения об ошибке теста."""
    return violations.head(MAX_REPORTED_ROWS).to_string(index=False)


# =============================================================================
#  1. ИНВАРИАНТЫ РАЗДАЧИ
# =============================================================================

def test_net_result_is_zero_sum_per_hand(connection: sqlite3.Connection) -> None:
    """Покер — игра с нулевой суммой: сколько один выиграл, столько другие проиграли.

    Рейк не взимается, поэтому сумма net_result по каждой раздаче строго равна 0.
    Ненулевая сумма означала бы «созданные из воздуха» или потерянные фишки.
    """
    violations = fetch(connection, """
        SELECT hand_id, SUM(net_result) AS total_chips
        FROM hand_players
        GROUP BY hand_id
        HAVING SUM(net_result) <> 0
    """)
    assert violations.empty, f"Сумма фишек не равна нулю в {len(violations)} раздачах:\n{describe(violations)}"


def test_pot_equals_sum_of_action_amounts(connection: sqlite3.Connection) -> None:
    """Банк раздачи должен совпадать с суммой всех вложенных в неё фишек."""
    violations = fetch(connection, """
        SELECT h.hand_id, h.pot_size, COALESCE(SUM(a.amount), 0) AS actions_total
        FROM hands        AS h
        LEFT JOIN actions AS a ON a.hand_id = h.hand_id
        GROUP BY h.hand_id
        HAVING h.pot_size <> COALESCE(SUM(a.amount), 0)
    """)
    assert violations.empty, f"Банк не сходится с суммой ставок в {len(violations)} раздачах:\n{describe(violations)}"


def test_every_hand_has_six_distinct_seats(connection: sqlite3.Connection) -> None:
    """За 6-max столом ровно 6 игроков, и каждая позиция занята один раз."""
    violations = fetch(connection, f"""
        SELECT hand_id,
               COUNT(*)                     AS seats,
               COUNT(DISTINCT position_id)  AS distinct_positions
        FROM hand_players
        GROUP BY hand_id
        HAVING COUNT(*) <> {SEATS_PER_HAND} OR COUNT(DISTINCT position_id) <> {SEATS_PER_HAND}
    """)
    assert violations.empty, f"Некорректный состав стола в {len(violations)} раздачах:\n{describe(violations)}"


def test_nobody_loses_more_than_starting_stack(connection: sqlite3.Connection) -> None:
    """Игрок не может проиграть больше, чем принёс за стол."""
    violations = fetch(connection, """
        SELECT hand_id, player_id, net_result, starting_stack
        FROM hand_players
        WHERE net_result < -starting_stack
    """)
    assert violations.empty, f"Проигрыш больше стека в {len(violations)} строках:\n{describe(violations)}"


# =============================================================================
#  2. АДЕКВАТНОСТЬ МЕТРИК ВИТРИНЫ
# =============================================================================

@pytest.mark.parametrize("metric", ["vpip_pct", "pfr_pct", "three_bet_pct", "wtsd_pct"])
def test_percentage_metrics_within_bounds(player_stats: pd.DataFrame, metric: str) -> None:
    """Любой процент лежит в [0, 100].

    NULL допустим: three_bet_pct и wtsd_pct не определены, если у игрока не было
    ни одной возможности 3-бета / он ни разу не увидел флоп.
    """
    low, high = PERCENTAGE_BOUNDS
    values = player_stats[metric].dropna()
    out_of_range = player_stats.loc[~player_stats[metric].between(low, high) & player_stats[metric].notna(),
                                    ["player_name", metric]]
    assert not values.empty, f"Метрика {metric} не заполнена ни у одного игрока"
    assert out_of_range.empty, f"{metric} вне диапазона [{low}, {high}]:\n{describe(out_of_range)}"


def test_pfr_never_exceeds_vpip(player_stats: pd.DataFrame) -> None:
    """Рейз префлоп — частный случай добровольного вложения денег, поэтому PFR ≤ VPIP.

    Нарушение означало бы ошибку в определении флагов в v_hand_player_facts:
    например, что рейз не попал в VPIP.
    """
    violations = player_stats.loc[player_stats["pfr_pct"] > player_stats["vpip_pct"],
                                  ["player_name", "vpip_pct", "pfr_pct"]]
    assert violations.empty, f"PFR больше VPIP у {len(violations)} игроков:\n{describe(violations)}"


def test_aggregates_match_row_counts(connection: sqlite3.Connection, player_stats: pd.DataFrame) -> None:
    """Число раздач в витрине совпадает с числом строк игрока в hand_players."""
    expected = fetch(connection, """
        SELECT player_id, COUNT(*) AS hands_in_table
        FROM hand_players
        GROUP BY player_id
    """)
    merged = player_stats[["player_id", "player_name", "hands"]].merge(expected, on="player_id", how="outer")
    violations = merged.loc[merged["hands"] != merged["hands_in_table"]]
    assert violations.empty, f"Витрина расходится с таблицей у {len(violations)} игроков:\n{describe(violations)}"


# =============================================================================
#  3. ЦЕЛОСТНОСТЬ СВЯЗЕЙ
# =============================================================================

def test_actions_belong_to_seated_players(connection: sqlite3.Connection) -> None:
    """Действовать может только игрок, физически сидевший в этой раздаче.

    За это отвечает составной внешний ключ actions(hand_id, player_id) →
    hand_players, но тест защищает от «тихого» расхождения: FK в SQLite
    проверяются только при включённом PRAGMA foreign_keys.
    """
    violations = fetch(connection, """
        SELECT a.action_id, a.hand_id, a.player_id
        FROM actions           AS a
        LEFT JOIN hand_players AS hp
               ON hp.hand_id = a.hand_id AND hp.player_id = a.player_id
        WHERE hp.player_id IS NULL
    """)
    assert violations.empty, (
        f"{len(violations)} действий принадлежат игрокам вне раздачи:\n{describe(violations)}"
    )


def test_winner_is_seated_in_the_hand(connection: sqlite3.Connection) -> None:
    """Банк забирает участник раздачи, а не посторонний игрок."""
    violations = fetch(connection, """
        SELECT h.hand_id, h.winner_player_id
        FROM hands             AS h
        LEFT JOIN hand_players AS hp
               ON hp.hand_id = h.hand_id AND hp.player_id = h.winner_player_id
        WHERE hp.player_id IS NULL
    """)
    assert violations.empty, f"Победитель не сидел за столом в {len(violations)} раздачах:\n{describe(violations)}"


def test_no_foreign_key_violations(connection: sqlite3.Connection) -> None:
    """Штатная проверка SQLite: все внешние ключи ссылаются на существующие строки."""
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    assert not violations, f"Нарушения внешних ключей: {violations[:MAX_REPORTED_ROWS]}"


def test_amount_matches_action_type(connection: sqlite3.Connection) -> None:
    """fold и check — всегда с нулевой суммой, call/bet/raise — всегда с положительной."""
    violations = fetch(connection, """
        SELECT a.action_id, t.action_name, a.amount
        FROM actions      AS a
        JOIN action_types AS t ON t.action_type_id = a.action_type_id
        WHERE (t.requires_amount = 1) <> (a.amount > 0)
    """)
    assert violations.empty, f"Сумма не соответствует типу действия в {len(violations)} строках:\n{describe(violations)}"


# =============================================================================
#  4. КОНТРАКТ СХЕМЫ (негативные тесты)
# =============================================================================
# Проверяем, что ограничения БД реально работают: попытка записать некорректные
# данные должна падать. Все вставки выполняются в транзакции и откатываются,
# поэтому содержимое базы не меняется.

@pytest.fixture
def rollback_cursor(connection: sqlite3.Connection) -> Iterator[sqlite3.Cursor]:
    """Курсор, все изменения которого гарантированно откатываются после теста."""
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        connection.rollback()


def test_schema_rejects_action_from_unseated_player(
    connection: sqlite3.Connection, rollback_cursor: sqlite3.Cursor
) -> None:
    """Составной FK не даёт записать действие игрока, которого нет в раздаче."""
    hand_id, seq = connection.execute(
        "SELECT hand_id, MAX(action_seq) + 1 FROM actions GROUP BY hand_id LIMIT 1"
    ).fetchone()
    stranger_id = connection.execute("""
        SELECT player_id FROM players
        WHERE player_id NOT IN (SELECT player_id FROM hand_players WHERE hand_id = ?)
        LIMIT 1
    """, (hand_id,)).fetchone()[0]

    with pytest.raises(sqlite3.IntegrityError):
        rollback_cursor.execute(
            "INSERT INTO actions (hand_id, player_id, street_id, action_type_id, action_seq, amount) "
            "VALUES (?, ?, 1, 2, ?, 0)",
            (hand_id, stranger_id, seq),
        )


def test_schema_rejects_fold_with_amount(
    connection: sqlite3.Connection, rollback_cursor: sqlite3.Cursor
) -> None:
    """Триггер не даёт записать fold с ненулевой суммой."""
    hand_id, player_id, seq = connection.execute("""
        SELECT a.hand_id, a.player_id, MAX(a.action_seq) + 1
        FROM actions AS a
        GROUP BY a.hand_id
        LIMIT 1
    """).fetchone()

    with pytest.raises(sqlite3.IntegrityError):
        rollback_cursor.execute(
            "INSERT INTO actions (hand_id, player_id, street_id, action_type_id, action_seq, amount) "
            "VALUES (?, ?, 1, 2, ?, 500)",   # action_type_id = 2 → fold
            (hand_id, player_id, seq),
        )


def test_schema_rejects_wrong_type_in_strict_table(rollback_cursor: sqlite3.Cursor) -> None:
    """STRICT-таблица отвергает текст там, где объявлен INTEGER."""
    with pytest.raises(sqlite3.IntegrityError):
        rollback_cursor.execute("INSERT INTO players (player_id, player_name) VALUES ('not-an-id', 'Ghost')")
