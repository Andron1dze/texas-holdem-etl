"""
data_loader.py — ETL-этап проекта «Texas Hold'em ETL & Analytics Pipeline».

Пайплайн:
    1. EXTRACT   — генерация синтетического, но логически связного лога
                   6-max раздач (NL Hold'em, блайнды 50/100, стеки 100 BB).
    2. TRANSFORM — разбиение плоского лога на нормализованные таблицы
                   (players, hands, hand_players, actions) и замена
                   натуральных ключей на суррогатные id из справочников БД.
    3. VALIDATE  — проверка бизнес-инвариантов до загрузки
                   (zero-sum, банк = сумма ставок, 6 мест за столом и т.д.).
    4. LOAD      — загрузка DataFrame'ов в SQLite (`poker_analytics.db`)
                   в порядке зависимостей внешних ключей + пост-проверка.

Модель симуляции (упрощённая, но правдоподобная):
    * Пул из N игроков четырёх архетипов (Nit / TAG / LAG / Fish), у каждого —
      индивидуальные частоты (ширина диапазона, 3-бет, агрессия, лимпы).
    * Каждая раздача — случайный стол из пула (как в fast-fold покере).
    * Сила префлоп-руки — перцентиль формулы Чена (векторно для всех раздач).
    * Решения зависят от силы руки, позиции, типа игрока и размера банка
      (pot odds). Постфлоп-сила — случайное блуждание от префлоп-силы,
      победитель на шоудауне — игрок с максимальной силой на ривере.

Запуск из корня репозитория:
    python src/data_loader.py --hands 1000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger("data_loader")

# =============================================================================
#  КОНСТАНТЫ ПРЕДМЕТНОЙ ОБЛАСТИ
# =============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

SEATS: Final[int] = 6
POSITIONS: Final[tuple[str, ...]] = ("UTG", "HJ", "CO", "BTN", "SB", "BB")     # порядок хода префлоп
POSTFLOP_ORDER: Final[tuple[str, ...]] = ("SB", "BB", "UTG", "HJ", "CO", "BTN")
BLINDS: Final[frozenset[str]] = frozenset({"SB", "BB"})
STREETS: Final[tuple[str, ...]] = ("preflop", "flop", "turn", "river")
POSTFLOP_STREETS: Final[tuple[str, ...]] = STREETS[1:]

RANKS: Final[str] = "23456789TJQKA"   # индекс ранга = card // 4
SUITS: Final[str] = "cdhs"            # индекс масти = card % 4
DECK_SIZE: Final[int] = 52

AGGRESSIVE_ACTIONS: Final[frozenset[str]] = frozenset({"bet", "raise", "3-bet", "4-bet"})
CHIP_ACTIONS: Final[frozenset[str]] = AGGRESSIVE_ACTIONS | {"call", "post_blind"}

# --- Префлоп: множитель ширины диапазона по позициям (поздняя позиция → шире)
POSITION_RANGE_MULTIPLIER: Final[dict[str, float]] = {
    "UTG": 0.65, "HJ": 0.80, "CO": 1.00, "BTN": 1.35, "SB": 1.00, "BB": 1.30,
}
OPEN_SIZE_BB: Final[float] = 2.5          # стандартный опен-рейз
OPEN_SIZE_SB_BB: Final[float] = 3.0       # опен из SB (без позиции)
THREE_BET_IP_MULT: Final[float] = 3.0     # 3-бет в позиции: x3 от опена
THREE_BET_OOP_MULT: Final[float] = 4.0    # 3-бет из блайндов: x4
FOUR_BET_MULT: Final[float] = 2.3         # 4-бет: x2.3 от 3-бета
FLAT_CALL_SHARE: Final[float] = 0.6       # доля диапазона, которым коллируют опен
BB_ISO_RAISE_THRESHOLD: Final[float] = 0.85
FOUR_BET_THRESHOLD: Final[float] = 0.97   # топ-3% рук
CALL_3BET_THRESHOLD: Final[float] = 0.88
CALL_4BET_THRESHOLD: Final[float] = 0.95

# --- Постфлоп
STRENGTH_PERSISTENCE: Final[tuple[float, float, float]] = (0.60, 0.75, 0.75)  # flop, turn, river
VALUE_BET_THRESHOLD: Final[float] = 0.70
BLUFF_CEILING: Final[float] = 0.50
BLUFF_FREQUENCY: Final[float] = 0.25
POSTFLOP_RAISE_THRESHOLD: Final[float] = 0.85
POSTFLOP_RAISE_MULT: Final[float] = 3.0
CALL_MARGIN: Final[float] = 0.45
BET_SIZE_FRACTIONS: Final[npt.NDArray[np.float64]] = np.array([0.33, 0.50, 0.66, 0.75, 1.00])

# --- Формула Чена: очки старшей карты для рангов 2..A
CHEN_HIGH_CARD_POINTS: Final[npt.NDArray[np.float64]] = np.array(
    [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10], dtype=np.float64
)
RANK_QUEEN: Final[int] = RANKS.index("Q")

# --- Игроки
NICKNAME_PREFIXES: Final[tuple[str, ...]] = (
    "Lucky", "Silent", "Wild", "Cold", "Big", "Sneaky", "Iron", "Tilted", "Royal", "Dark",
)
NICKNAME_SUFFIXES: Final[tuple[str, ...]] = (
    "Shark", "Fish", "Ace", "River", "Bluff", "Rock", "Maniac", "Donk",
)

# Порядок таблиц при загрузке — от родителей к потомкам (FK)
LOAD_ORDER: Final[tuple[str, ...]] = ("players", "hands", "hand_players", "actions")

HANDS_COLUMNS: Final[list[str]] = [
    "hand_id", "played_at", "small_blind", "big_blind", "pot_size",
    "winner_player_id", "final_street_id", "went_to_showdown",
]
HAND_PLAYERS_COLUMNS: Final[list[str]] = [
    "hand_id", "player_id", "position_id", "hole_cards", "starting_stack", "net_result",
]
ACTIONS_COLUMNS: Final[list[str]] = [
    "action_id", "hand_id", "player_id", "street_id", "action_type_id",
    "action_seq", "amount", "is_all_in",
]


class DataValidationError(Exception):
    """Нарушен бизнес-инвариант данных — загрузка в БД прерывается."""


# =============================================================================
#  КОНФИГУРАЦИЯ И МОДЕЛИ
# =============================================================================

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Параметры генерации и пути ввода-вывода."""

    n_hands: int = 1_000
    n_players: int = 30
    seed: int = 42
    small_blind: int = 50
    big_blind: int = 100
    starting_stack_bb: int = 100
    session_start: datetime = datetime(2026, 9, 1, 18, 0, 0)
    mean_seconds_between_hands: float = 75.0
    db_path: Path = PROJECT_ROOT / "data" / "poker_analytics.db"
    schema_path: Path = PROJECT_ROOT / "sql" / "database_setup.sql"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"

    def __post_init__(self) -> None:
        max_players = len(NICKNAME_PREFIXES) * len(NICKNAME_SUFFIXES)
        if self.n_hands < 1:
            raise ValueError("n_hands must be positive")
        if not SEATS <= self.n_players <= max_players:
            raise ValueError(f"n_players must be in [{SEATS}, {max_players}]")
        if self.big_blind <= self.small_blind:
            raise ValueError("big_blind must exceed small_blind")

    @property
    def starting_stack(self) -> int:
        return self.starting_stack_bb * self.big_blind


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """Скрытые (ground truth) параметры стиля игрока."""

    name: str
    archetype: str
    vpip_base: float        # базовая ширина диапазона (итоговый VPIP ниже из-за рейзов впереди)
    three_bet: float        # доля рук, с которыми игрок 3-бетит
    aggression: float       # склонность ставить/рейзить постфлоп
    limp_rate: float        # вероятность лимпа вместо опен-рейза
    call_looseness: float   # сдвиг порога колла постфлоп (>0 — «коллинг-станция»)


# Средние параметры архетипов: (vpip_base, three_bet, aggression, limp_rate, call_looseness)
ARCHETYPES: Final[dict[str, tuple[float, float, float, float, float]]] = {
    "Nit":  (0.14, 0.03, 0.40, 0.02, -0.08),
    "TAG":  (0.22, 0.07, 0.60, 0.02, 0.00),
    "LAG":  (0.32, 0.11, 0.80, 0.03, 0.03),
    "Fish": (0.45, 0.02, 0.25, 0.55, 0.15),
}
ARCHETYPE_WEIGHTS: Final[npt.NDArray[np.float64]] = np.array([0.20, 0.35, 0.20, 0.25])
PARAM_NOISE_SD: Final[npt.NDArray[np.float64]] = np.array([0.03, 0.015, 0.07, 0.05, 0.03])
PARAM_LOWER: Final[npt.NDArray[np.float64]] = np.array([0.08, 0.01, 0.05, 0.00, -0.20])
PARAM_UPPER: Final[npt.NDArray[np.float64]] = np.array([0.70, 0.20, 0.95, 0.80, 0.30])


@dataclass(frozen=True, slots=True)
class Decision:
    """Решение игрока. target — до какой суммы довести вклад на текущей улице."""

    action: str
    target: int = 0


@dataclass(slots=True)
class Seat:
    """Изменяемое состояние игрока внутри одной раздачи."""

    player: PlayerProfile
    position: str
    hole_cards: str
    preflop_strength: float
    street_strength: dict[str, float]
    starting_stack: int
    stack: int
    street_committed: int = 0
    total_committed: int = 0
    is_active: bool = True          # False после фолда

    @property
    def can_act(self) -> bool:
        """Игрок в раздаче и у него остались фишки (не в олл-ине)."""
        return self.is_active and self.stack > 0


@dataclass(frozen=True, slots=True)
class Lookups:
    """Справочники БД: натуральный ключ → суррогатный id."""

    positions: dict[str, int]
    streets: dict[str, int]
    action_types: dict[str, int]
    amount_required_ids: frozenset[int]


# =============================================================================
#  КАРТЫ И СИЛА РУК (векторизовано на NumPy)
# =============================================================================

def chen_score(card_a: npt.NDArray[np.int_], card_b: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Формула Билла Чена для стартовой руки. Работает с массивами любой формы."""
    rank_a, rank_b = card_a // 4, card_b // 4
    high, low = np.maximum(rank_a, rank_b), np.minimum(rank_a, rank_b)
    is_pair = high == low
    is_suited = (card_a % 4) == (card_b % 4)
    gap = high - low - 1

    score = CHEN_HIGH_CARD_POINTS[high]
    score = np.where(is_pair, np.maximum(5.0, 2.0 * score), score)
    score = score + 2.0 * is_suited

    gap_penalty = np.select([gap <= 0, gap == 1, gap == 2, gap == 3], [0, 1, 2, 4], default=5)
    score = np.where(is_pair, score, score - gap_penalty)

    connector_bonus = (~is_pair) & (gap <= 1) & (high < RANK_QUEEN)
    return np.ceil(score + connector_bonus)


@lru_cache(maxsize=1)
def _reference_chen_scores() -> npt.NDArray[np.float64]:
    """Отсортированные очки Чена всех 1326 комбинаций — эталон для перцентилей."""
    card_a, card_b = np.triu_indices(DECK_SIZE, k=1)
    return np.sort(chen_score(card_a, card_b))


def preflop_strength(card_a: npt.NDArray[np.int_], card_b: npt.NDArray[np.int_]) -> npt.NDArray[np.float64]:
    """Перцентиль силы руки в [0, 1] (mid-rank для одинаковых очков). AA ≈ 1.0."""
    reference = _reference_chen_scores()
    scores = chen_score(card_a, card_b)
    left = np.searchsorted(reference, scores, side="left")
    right = np.searchsorted(reference, scores, side="right")
    return (left + right) / (2.0 * reference.size)


def simulate_street_strength(
    preflop: npt.NDArray[np.float64], rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Сила руки на flop/turn/river как случайное блуждание от префлоп-силы.

    Возвращает массив формы (*preflop.shape, 3). Сильные префлоп-руки чаще
    остаются сильными, но «доезды» возможны — как в реальной игре.
    """
    noise = rng.random((*preflop.shape, len(POSTFLOP_STREETS)))
    strengths: list[npt.NDArray[np.float64]] = []
    previous = preflop
    for street_idx, persistence in enumerate(STRENGTH_PERSISTENCE):
        current = persistence * previous + (1.0 - persistence) * noise[..., street_idx]
        strengths.append(current)
        previous = current
    return np.stack(strengths, axis=-1)


def card_labels(cards: npt.NDArray[np.int_]) -> npt.NDArray[np.str_]:
    """Кодирует пары карт (…, 2) в строки вида 'AhKs'."""
    ranks = np.array(list(RANKS))
    suits = np.array(list(SUITS))
    first, second = cards[..., 0], cards[..., 1]
    labels = np.char.add(ranks[first // 4], suits[first % 4])
    labels = np.char.add(labels, ranks[second // 4])
    return np.char.add(labels, suits[second % 4])


def deal_hole_cards(n_hands: int, rng: np.random.Generator) -> npt.NDArray[np.int_]:
    """Раздаёт по 2 карты 6 игрокам в каждой раздаче без повторов внутри колоды.

    Возвращает (n_hands, SEATS, 2); в каждой паре старшая карта идёт первой.
    """
    shuffled = rng.random((n_hands, DECK_SIZE)).argsort(axis=1)
    cards = shuffled[:, : 2 * SEATS].reshape(n_hands, SEATS, 2)
    return np.sort(cards, axis=-1)[..., ::-1]


# =============================================================================
#  ИГРОКИ, РАССАДКА, ВРЕМЯ
# =============================================================================

def build_player_pool(n_players: int, rng: np.random.Generator) -> list[PlayerProfile]:
    """Создаёт пул игроков: уникальные ники + параметры архетипа с шумом."""
    all_nicknames = np.array([p + s for p in NICKNAME_PREFIXES for s in NICKNAME_SUFFIXES])
    names = rng.choice(all_nicknames, size=n_players, replace=False)
    archetypes = rng.choice(list(ARCHETYPES), size=n_players, p=ARCHETYPE_WEIGHTS)

    means = np.array([ARCHETYPES[a] for a in archetypes])                 # (n, 5)
    params = np.clip(means + rng.normal(0.0, PARAM_NOISE_SD, means.shape), PARAM_LOWER, PARAM_UPPER)

    return [
        PlayerProfile(str(name), str(archetype), *map(float, row))
        for name, archetype, row in zip(names, archetypes, params)
    ]


def assign_seats(n_hands: int, n_players: int, rng: np.random.Generator) -> npt.NDArray[np.int_]:
    """Случайный стол из пула на каждую раздачу: (n_hands, SEATS) индексов игроков."""
    return rng.random((n_hands, n_players)).argsort(axis=1)[:, :SEATS]


def build_timestamps(config: PipelineConfig, rng: np.random.Generator) -> list[str]:
    """Время начала раздач: пуассоновский поток с заданным средним интервалом."""
    gaps = rng.exponential(config.mean_seconds_between_hands, config.n_hands).round()
    moments = pd.Timestamp(config.session_start) + pd.to_timedelta(np.cumsum(gaps), unit="s")
    return moments.strftime("%Y-%m-%d %H:%M:%S").tolist()


# =============================================================================
#  СИМУЛЯТОР ОДНОЙ РАЗДАЧИ
# =============================================================================

class HandSimulator:
    """Проигрывает раздачу по улицам и возвращает плоский лог действий."""

    def __init__(self, config: PipelineConfig, rng: np.random.Generator) -> None:
        self._cfg = config
        self._rng = rng
        self._seats: list[Seat] = []
        self._records: list[dict[str, object]] = []
        self._limpers = 0
        self._hand_id = 0
        self._played_at = ""

    # --- публичный API --------------------------------------------------------

    def simulate(self, hand_id: int, played_at: str, seats: list[Seat]) -> list[dict[str, object]]:
        """Проигрывает раздачу целиком. seats упорядочены как POSITIONS."""
        self._hand_id, self._played_at, self._seats = hand_id, played_at, seats
        self._records, self._limpers = [], 0
        by_position = {seat.position: seat for seat in seats}

        self._post_blind(by_position["SB"], self._cfg.small_blind)
        self._post_blind(by_position["BB"], self._cfg.big_blind)
        self._betting_round("preflop", [by_position[p] for p in POSITIONS], self._cfg.big_blind)

        final_street = "preflop"
        postflop_order = [by_position[p] for p in POSTFLOP_ORDER]
        for street in POSTFLOP_STREETS:
            if self._active_count() == 1:
                break
            final_street = street
            for seat in seats:
                seat.street_committed = 0
            # Если торговаться некому (все, кроме одного, в олл-ине) — карты просто открываются
            if sum(seat.can_act for seat in seats) >= 2:
                self._betting_round(street, postflop_order, opening_bet=0)

        winner, went_to_showdown = self._resolve_winner()
        for record in self._records:
            record.update(
                winner_name=winner.player.name,
                final_street=final_street,
                went_to_showdown=int(went_to_showdown),
            )
        return self._records

    # --- механика торговли ----------------------------------------------------

    def _betting_round(self, street: str, order: list[Seat], opening_bet: int) -> None:
        """Круг торговли: ходят по очереди, пока все не уравняют последнюю ставку."""
        current_bet, raises = opening_bet, 0
        pending = [seat for seat in order if seat.can_act]

        while pending and self._active_count() > 1:
            seat = pending.pop(0)
            if not seat.can_act:
                continue
            decision = self._legalize(seat, self._decide(street, seat, current_bet, raises), current_bet)
            self._apply(street, seat, decision)

            if street == "preflop" and raises == 0 and decision.action == "call":
                self._limpers += 1
            if decision.action in AGGRESSIVE_ACTIONS:
                current_bet, raises = seat.street_committed, raises + 1
                # После агрессии все остальные активные игроки обязаны ответить
                idx = order.index(seat)
                pending = [s for s in order[idx + 1:] + order[:idx] if s.can_act]

    def _post_blind(self, seat: Seat, amount: int) -> None:
        self._apply("preflop", seat, Decision("post_blind", amount))

    def _apply(self, street: str, seat: Seat, decision: Decision) -> None:
        """Применяет решение к состоянию стола и пишет строку в лог."""
        if decision.action == "fold":
            seat.is_active = False
        amount = 0
        if decision.action in CHIP_ACTIONS:
            amount = max(0, min(decision.target - seat.street_committed, seat.stack))
        seat.stack -= amount
        seat.street_committed += amount
        seat.total_committed += amount
        self._record(street, seat, decision.action, amount)

    @staticmethod
    def _legalize(seat: Seat, decision: Decision, current_bet: int) -> Decision:
        """Приводит «желание» игрока к легальному действию по правилам."""
        to_call = current_bet - seat.street_committed
        max_total = seat.street_committed + seat.stack

        if decision.action in AGGRESSIVE_ACTIONS:
            target = min(decision.target, max_total)
            if target > current_bet:
                return Decision(decision.action, target)
            decision = Decision("call")                  # не хватает фишек на рейз
        if decision.action == "call":
            return Decision("call", current_bet) if to_call > 0 else Decision("check")
        if decision.action == "check" and to_call > 0:
            return Decision("fold")
        if decision.action == "fold" and to_call == 0:
            return Decision("check")                     # бесплатно не фолдят
        return decision

    # --- модели решений -------------------------------------------------------

    def _decide(self, street: str, seat: Seat, current_bet: int, raises: int) -> Decision:
        if street == "preflop":
            return self._decide_preflop(seat, current_bet, raises)
        return self._decide_postflop(street, seat, current_bet, raises)

    def _decide_preflop(self, seat: Seat, current_bet: int, raises: int) -> Decision:
        """Префлоп: диапазоны зависят от позиции, стиля и количества рейзов."""
        strength, profile, bb = seat.preflop_strength, seat.player, self._cfg.big_blind
        range_width = profile.vpip_base * POSITION_RANGE_MULTIPLIER[seat.position]
        to_call = current_bet - seat.street_committed

        if raises == 0:                                   # банк не открыт (возможны лимперы)
            if strength < 1.0 - range_width:
                return Decision("fold")
            if to_call == 0 and strength < BB_ISO_RAISE_THRESHOLD:
                return Decision("check")                  # BB: опция чека
            if to_call > 0 and self._rng.random() < profile.limp_rate:
                return Decision("call")                   # лимп
            open_bb = OPEN_SIZE_SB_BB if seat.position == "SB" else OPEN_SIZE_BB
            return Decision("raise", int((open_bb + self._limpers) * bb))

        if raises == 1:                                   # против опен-рейза
            if strength >= 1.0 - profile.three_bet:
                mult = THREE_BET_OOP_MULT if seat.position in BLINDS else THREE_BET_IP_MULT
                return Decision("3-bet", int(current_bet * mult))
            if strength >= 1.0 - range_width * FLAT_CALL_SHARE:
                return Decision("call")
            return Decision("fold")

        if raises == 2:                                   # против 3-бета
            if strength >= FOUR_BET_THRESHOLD:
                return Decision("4-bet", int(current_bet * FOUR_BET_MULT))
            if strength >= CALL_3BET_THRESHOLD - 0.1 * profile.call_looseness:
                return Decision("call")
            return Decision("fold")

        return Decision("call") if strength >= CALL_4BET_THRESHOLD else Decision("fold")

    def _decide_postflop(self, street: str, seat: Seat, current_bet: int, raises: int) -> Decision:
        """Постфлоп: вэлью-ставки, блефы, рейзы и коллы по шансам банка."""
        strength, profile = seat.street_strength[street], seat.player
        pot = self._pot()
        to_call = current_bet - seat.street_committed

        if to_call == 0:
            wants_value = (strength >= VALUE_BET_THRESHOLD
                           and self._rng.random() < 0.35 + 0.6 * profile.aggression)
            wants_bluff = (strength < BLUFF_CEILING
                           and self._rng.random() < BLUFF_FREQUENCY * profile.aggression)
            if wants_value or wants_bluff:
                return Decision("bet", self._bet_size(pot))
            return Decision("check")

        if raises == 1 and strength >= POSTFLOP_RAISE_THRESHOLD and self._rng.random() < profile.aggression:
            return Decision("raise", int(current_bet * POSTFLOP_RAISE_MULT))

        pot_odds = to_call / (pot + to_call)
        call_threshold = pot_odds + CALL_MARGIN - profile.call_looseness
        return Decision("call") if strength >= call_threshold else Decision("fold")

    # --- вспомогательные ------------------------------------------------------

    def _bet_size(self, pot: int) -> int:
        """Ставка как доля банка, округлённая до малого блайнда (не меньше BB)."""
        fraction = float(self._rng.choice(BET_SIZE_FRACTIONS))
        sb = self._cfg.small_blind
        return max(self._cfg.big_blind, int(round(pot * fraction / sb)) * sb)

    def _pot(self) -> int:
        return sum(seat.total_committed for seat in self._seats)

    def _active_count(self) -> int:
        return sum(seat.is_active for seat in self._seats)

    def _resolve_winner(self) -> tuple[Seat, bool]:
        """Последний оставшийся игрок или лучшая рука на ривере (шоудаун)."""
        active = [seat for seat in self._seats if seat.is_active]
        if len(active) == 1:
            return active[0], False
        return max(active, key=lambda seat: seat.street_strength["river"]), True

    def _record(self, street: str, seat: Seat, action: str, amount: int) -> None:
        self._records.append({
            "hand_id": self._hand_id,
            "played_at": self._played_at,
            "small_blind": self._cfg.small_blind,
            "big_blind": self._cfg.big_blind,
            "action_seq": len(self._records) + 1,
            "street": street,
            "player_name": seat.player.name,
            "position": seat.position,
            "hole_cards": seat.hole_cards,
            "starting_stack": seat.starting_stack,
            "action": action,
            "amount": amount,
            "is_all_in": int(amount > 0 and seat.stack == 0),
        })


# =============================================================================
#  1. EXTRACT — генерация «сырого» лога
# =============================================================================

def generate_raw_hand_log(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Генерирует плоский (денормализованный) лог действий и профили игроков.

    Всё, что не зависит от хода торговли, считается векторно для всех
    раздач сразу; последовательная часть — только сама торговля.
    """
    rng = np.random.default_rng(config.seed)
    profiles = build_player_pool(config.n_players, rng)
    seating = assign_seats(config.n_hands, config.n_players, rng)            # (n, 6)
    cards = deal_hole_cards(config.n_hands, rng)                             # (n, 6, 2)
    strength = preflop_strength(cards[..., 0], cards[..., 1])                # (n, 6)
    street_strength = simulate_street_strength(strength, rng)                # (n, 6, 3)
    labels = card_labels(cards)                                              # (n, 6)
    timestamps = build_timestamps(config, rng)

    simulator = HandSimulator(config, rng)
    records: list[dict[str, object]] = []
    for h in range(config.n_hands):
        seats = [
            Seat(
                player=profiles[seating[h, i]],
                position=POSITIONS[i],
                hole_cards=str(labels[h, i]),
                preflop_strength=float(strength[h, i]),
                street_strength=dict(zip(POSTFLOP_STREETS, street_strength[h, i].tolist())),
                starting_stack=config.starting_stack,
                stack=config.starting_stack,
            )
            for i in range(SEATS)
        ]
        records.extend(simulator.simulate(hand_id=h + 1, played_at=timestamps[h], seats=seats))

    raw_log = pd.DataFrame.from_records(records)
    profiles_df = pd.DataFrame([asdict(p) for p in profiles]).sort_values("name", ignore_index=True)
    logger.info("Extract: %d hands, %d actions, %d players",
                config.n_hands, len(raw_log), len(profiles_df))
    return raw_log, profiles_df


def persist_raw_layer(raw_log: pd.DataFrame, profiles: pd.DataFrame, raw_dir: Path) -> None:
    """Сохраняет «сырой» слой в CSV — источник правды для повторного Transform."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_log.to_csv(raw_dir / "hand_log.csv", index=False)
    profiles.to_csv(raw_dir / "player_profiles.csv", index=False)   # ground truth для валидации аналитики
    logger.info("Raw layer saved to %s", raw_dir)


# =============================================================================
#  БАЗА ДАННЫХ: создание схемы и справочники
# =============================================================================

def create_database(db_path: Path, schema_path: Path) -> sqlite3.Connection:
    """Пересоздаёт файл БД и выполняет DDL. Идемпотентно: каждый запуск — с нуля."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    logger.info("Schema created: %s", db_path)
    return conn


def read_lookups(conn: sqlite3.Connection) -> Lookups:
    """Читает справочники из БД — id не дублируются в Python-коде."""
    def as_mapping(query: str) -> dict[str, int]:
        return {str(name): int(key) for name, key in conn.execute(query)}

    return Lookups(
        positions=as_mapping("SELECT position_code, position_id FROM positions"),
        streets=as_mapping("SELECT street_name, street_id FROM streets"),
        action_types=as_mapping("SELECT action_name, action_type_id FROM action_types"),
        amount_required_ids=frozenset(
            int(row[0]) for row in
            conn.execute("SELECT action_type_id FROM action_types WHERE requires_amount = 1")
        ),
    )


# =============================================================================
#  2. TRANSFORM — нормализация
# =============================================================================

def build_players_table(raw_log: pd.DataFrame) -> pd.DataFrame:
    """Уникальные игроки с суррогатным ключом (детерминированно по алфавиту)."""
    players = (raw_log[["player_name"]].drop_duplicates()
               .sort_values("player_name", ignore_index=True))
    players.insert(0, "player_id", np.arange(1, len(players) + 1, dtype=np.int64))
    return players


def attach_surrogate_keys(raw_log: pd.DataFrame, players: pd.DataFrame, lookups: Lookups) -> pd.DataFrame:
    """Заменяет натуральные ключи (имена, коды) на id. Неизвестное значение — ошибка."""
    player_ids = dict(zip(players["player_name"], players["player_id"]))
    key_mappings: dict[str, tuple[str, dict[str, int]]] = {
        "player_id":        ("player_name",  player_ids),
        "winner_player_id": ("winner_name",  player_ids),
        "position_id":      ("position",     lookups.positions),
        "street_id":        ("street",       lookups.streets),
        "final_street_id":  ("final_street", lookups.streets),
        "action_type_id":   ("action",       lookups.action_types),
    }
    enriched = raw_log.copy()
    for target, (source, mapping) in key_mappings.items():
        enriched[target] = enriched[source].map(mapping)
        unknown = enriched.loc[enriched[target].isna(), source].unique()
        if len(unknown) > 0:
            raise DataValidationError(f"Unknown values in '{source}': {list(unknown)[:5]}")
        enriched[target] = enriched[target].astype(np.int64)
    return enriched


def build_hands_table(enriched: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на раздачу; банк = сумма всех вложенных фишек."""
    hands = enriched.groupby("hand_id", as_index=False, sort=True).agg(
        played_at=("played_at", "first"),
        small_blind=("small_blind", "first"),
        big_blind=("big_blind", "first"),
        pot_size=("amount", "sum"),
        winner_player_id=("winner_player_id", "first"),
        final_street_id=("final_street_id", "first"),
        went_to_showdown=("went_to_showdown", "first"),
    )
    return hands[HANDS_COLUMNS]


def build_hand_players_table(enriched: pd.DataFrame, hands: pd.DataFrame) -> pd.DataFrame:
    """Игрок в раздаче: позиция, карты и чистый результат (net = выигрыш − вложения)."""
    seats = enriched.groupby(["hand_id", "player_id"], as_index=False, sort=True).agg(
        position_id=("position_id", "first"),
        hole_cards=("hole_cards", "first"),
        starting_stack=("starting_stack", "first"),
        invested=("amount", "sum"),
    )
    seats = seats.merge(hands[["hand_id", "pot_size", "winner_player_id"]],
                        on="hand_id", how="left", validate="many_to_one")
    is_winner = seats["player_id"].eq(seats["winner_player_id"])
    seats["net_result"] = np.where(is_winner, seats["pot_size"], 0) - seats["invested"]
    return seats[HAND_PLAYERS_COLUMNS]


def build_actions_table(enriched: pd.DataFrame) -> pd.DataFrame:
    """Лог действий в хронологическом порядке с глобальным action_id."""
    actions = enriched.sort_values(["hand_id", "action_seq"], ignore_index=True)
    actions.insert(0, "action_id", np.arange(1, len(actions) + 1, dtype=np.int64))
    return actions[ACTIONS_COLUMNS]


def transform(raw_log: pd.DataFrame, lookups: Lookups) -> dict[str, pd.DataFrame]:
    """Плоский лог → 4 нормализованные таблицы, готовые к загрузке."""
    players = build_players_table(raw_log)
    enriched = attach_surrogate_keys(raw_log, players, lookups)
    hands = build_hands_table(enriched)
    tables = {
        "players": players,
        "hands": hands,
        "hand_players": build_hand_players_table(enriched, hands),
        "actions": build_actions_table(enriched),
    }
    logger.info("Transform: %s", {name: len(df) for name, df in tables.items()})
    return tables


# =============================================================================
#  3. VALIDATE — бизнес-инварианты до загрузки
# =============================================================================

def validate_tables(tables: dict[str, pd.DataFrame], lookups: Lookups) -> None:
    """Проверяет инварианты. Любое нарушение → DataValidationError (без загрузки)."""
    hands, seats, actions = tables["hands"], tables["hand_players"], tables["actions"]

    winners_seated = hands.merge(
        seats, left_on=["hand_id", "winner_player_id"], right_on=["hand_id", "player_id"], how="left"
    )["player_id"].notna()
    requires_amount = actions["action_type_id"].isin(lookups.amount_required_ids)
    chips_lost = seats["net_result"].clip(upper=0).abs()

    checks: dict[str, bool] = {
        "zero-sum: net results in every hand sum to 0":
            bool(seats.groupby("hand_id")["net_result"].sum().eq(0).all()),
        "pot equals sum of action amounts":
            bool(actions.groupby("hand_id")["amount"].sum()
                 .eq(hands.set_index("hand_id")["pot_size"]).all()),
        f"exactly {SEATS} seats per hand":
            bool(seats.groupby("hand_id").size().eq(SEATS).all()),
        "winner is seated in the hand":
            bool(winners_seated.all()),
        "nobody loses more than the starting stack":
            bool((chips_lost <= seats["starting_stack"]).all()),
        "amount > 0 exactly for chip actions":
            bool(((actions["amount"] > 0) == requires_amount).all()),
        "action_seq unique within hand":
            not actions.duplicated(["hand_id", "action_seq"]).any(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise DataValidationError("Validation failed: " + "; ".join(failed))
    logger.info("Validate: %d/%d checks passed", len(checks), len(checks))


# =============================================================================
#  4. LOAD — загрузка и пост-проверка
# =============================================================================

def load_tables(conn: sqlite3.Connection, tables: dict[str, pd.DataFrame]) -> None:
    """Дозаписывает DataFrame'ы в уже созданные STRICT-таблицы (схему не трогаем)."""
    for name in LOAD_ORDER:
        tables[name].to_sql(name, conn, if_exists="append", index=False, chunksize=10_000)
        logger.info("Load: %-13s %6d rows", name, len(tables[name]))
    conn.commit()


def verify_database(conn: sqlite3.Connection, expected_rows: dict[str, int]) -> None:
    """Сверяет количество строк и проверяет ссылочную целостность средствами SQLite."""
    fk_violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk_violations:
        raise DataValidationError(f"Foreign key violations: {fk_violations[:5]}")
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise DataValidationError(f"Integrity check failed: {integrity}")
    for table, expected in expected_rows.items():
        actual = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # имена — из константы
        if actual != expected:
            raise DataValidationError(f"{table}: expected {expected} rows, got {actual}")
    logger.info("Verify: row counts match, FK and integrity checks passed")


def log_summary(conn: sqlite3.Connection) -> None:
    """Короткая сводка по загруженным данным — санити-чек глазами."""
    summary = pd.read_sql_query(
        """
        SELECT s.street_name                                   AS final_street,
               COUNT(*)                                        AS hands,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS share_pct,
               ROUND(AVG(1.0 * h.pot_size / h.big_blind), 1)   AS avg_pot_bb
        FROM hands AS h
        JOIN streets AS s ON s.street_id = h.final_street_id
        GROUP BY s.street_id
        ORDER BY s.street_id
        """,
        conn,
    )
    logger.info("Hands by final street:\n%s", summary.to_string(index=False))


# =============================================================================
#  ОРКЕСТРАЦИЯ
# =============================================================================

def run_pipeline(config: PipelineConfig) -> dict[str, int]:
    """Полный прогон ETL. Возвращает количество загруженных строк по таблицам."""
    raw_log, profiles = generate_raw_hand_log(config)
    persist_raw_layer(raw_log, profiles, config.raw_dir)

    with closing(create_database(config.db_path, config.schema_path)) as conn:
        lookups = read_lookups(conn)
        tables = transform(raw_log, lookups)
        validate_tables(tables, lookups)
        load_tables(conn, tables)

        row_counts = {name: len(df) for name, df in tables.items()}
        verify_database(conn, row_counts)
        log_summary(conn)
    return row_counts


def parse_args() -> PipelineConfig:
    defaults = PipelineConfig()   # slots=True: значения по умолчанию берём из экземпляра
    parser = argparse.ArgumentParser(description="Generate and load synthetic Hold'em hands into SQLite.")
    parser.add_argument("--hands", type=int, default=defaults.n_hands, help="number of hands")
    parser.add_argument("--players", type=int, default=defaults.n_players, help="player pool size")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="random seed")
    parser.add_argument("--db", type=Path, default=defaults.db_path, help="SQLite file path")
    args = parser.parse_args()
    return PipelineConfig(n_hands=args.hands, n_players=args.players, seed=args.seed, db_path=args.db)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
