-- =============================================================================
--  Texas Hold'em ETL & Analytics Pipeline
--  database_setup.sql — DDL нормализованной схемы (SQLite >= 3.37)
-- -----------------------------------------------------------------------------
--  Принципы:
--    * STRICT-таблицы: SQLite отклоняет значения неподходящего типа
--      (по умолчанию SQLite допускает «мягкую» типизацию — здесь она выключена).
--    * Нормализация до 3НФ: повторяющиеся категории (позиции, улицы, типы
--      действий) вынесены в справочники, связь «игрок в раздаче» — в отдельную
--      таблицу hand_players.
--    * Все денежные величины хранятся в ФИШКАХ как INTEGER (без REAL),
--      чтобы исключить ошибки округления. Блайнды: 50 / 100.
--    * Целостность на уровне БД: PK, FK, UNIQUE, CHECK и триггер.
--
--  ER-модель:
--    players 1──* hand_players *──1 hands
--    hand_players 1──* actions        (составной FK: игрок обязан сидеть в раздаче)
--    positions / streets / action_types — справочники
-- =============================================================================

PRAGMA foreign_keys = ON;   -- в SQLite FK выключены по умолчанию (per-connection)

-- Пересоздание схемы с нуля (порядок обратный зависимостям) -------------------
DROP VIEW    IF EXISTS v_action_log;
DROP TRIGGER IF EXISTS trg_actions_amount_matches_type;
DROP TABLE   IF EXISTS actions;
DROP TABLE   IF EXISTS hand_players;
DROP TABLE   IF EXISTS hands;
DROP TABLE   IF EXISTS players;
DROP TABLE   IF EXISTS action_types;
DROP TABLE   IF EXISTS streets;
DROP TABLE   IF EXISTS positions;


-- =============================================================================
--  1. СПРАВОЧНИКИ (reference data)
-- =============================================================================

-- Позиции за 6-max столом. Порядок хода различается на префлопе и постфлопе.
CREATE TABLE positions (
    position_id     INTEGER PRIMARY KEY,
    position_code   TEXT    NOT NULL UNIQUE,
    position_name   TEXT    NOT NULL,
    preflop_order   INTEGER NOT NULL UNIQUE CHECK (preflop_order  BETWEEN 1 AND 6),
    postflop_order  INTEGER NOT NULL UNIQUE CHECK (postflop_order BETWEEN 1 AND 6)
) STRICT;

INSERT INTO positions (position_id, position_code, position_name, preflop_order, postflop_order) VALUES
    (1, 'UTG', 'Under the Gun', 1, 3),
    (2, 'HJ',  'Hijack',        2, 4),
    (3, 'CO',  'Cutoff',        3, 5),
    (4, 'BTN', 'Button',        4, 6),
    (5, 'SB',  'Small Blind',   5, 1),
    (6, 'BB',  'Big Blind',     6, 2);

-- Улицы торговли. street_id одновременно задаёт хронологический порядок.
CREATE TABLE streets (
    street_id    INTEGER PRIMARY KEY,
    street_name  TEXT    NOT NULL UNIQUE
                         CHECK (street_name IN ('preflop', 'flop', 'turn', 'river')),
    board_cards  INTEGER NOT NULL CHECK (board_cards IN (0, 3, 4, 5))
) STRICT;

INSERT INTO streets (street_id, street_name, board_cards) VALUES
    (1, 'preflop', 0),
    (2, 'flop',    3),
    (3, 'turn',    4),
    (4, 'river',   5);

-- Типы действий. Флаги упрощают расчёт покерных метрик в SQL:
--   is_voluntary  — добровольное вложение фишек (для VPIP);
--   is_aggressive — ставка/рейз (для PFR, Aggression Factor);
--   requires_amount — действие обязано иметь amount > 0.
CREATE TABLE action_types (
    action_type_id   INTEGER PRIMARY KEY,
    action_name      TEXT    NOT NULL UNIQUE,
    is_voluntary     INTEGER NOT NULL CHECK (is_voluntary    IN (0, 1)),
    is_aggressive    INTEGER NOT NULL CHECK (is_aggressive   IN (0, 1)),
    requires_amount  INTEGER NOT NULL CHECK (requires_amount IN (0, 1))
) STRICT;

INSERT INTO action_types (action_type_id, action_name, is_voluntary, is_aggressive, requires_amount) VALUES
    (1, 'post_blind', 0, 0, 1),
    (2, 'fold',       0, 0, 0),
    (3, 'check',      0, 0, 0),
    (4, 'call',       1, 0, 1),
    (5, 'bet',        1, 1, 1),
    (6, 'raise',      1, 1, 1),   -- префлоп: опен-рейз; постфлоп: рейз на ставку
    (7, '3-bet',      1, 1, 1),   -- только префлоп
    (8, '4-bet',      1, 1, 1);   -- только префлоп


-- =============================================================================
--  2. СУЩНОСТИ (core entities)
-- =============================================================================

CREATE TABLE players (
    player_id    INTEGER PRIMARY KEY,
    player_name  TEXT    NOT NULL UNIQUE
                         CHECK (length(trim(player_name)) BETWEEN 1 AND 50)
) STRICT;

-- Одна строка = одна раздача.
-- pot_size — итоговый банк (сумма всех вложенных фишек, рейк не взимается).
-- winner_player_id — забравший банк игрок (сплит-поты в симуляции не моделируются).
CREATE TABLE hands (
    hand_id           INTEGER PRIMARY KEY,
    played_at         TEXT    NOT NULL
                              CHECK (played_at = datetime(played_at)),  -- 'YYYY-MM-DD HH:MM:SS'
    small_blind       INTEGER NOT NULL CHECK (small_blind > 0),
    big_blind         INTEGER NOT NULL CHECK (big_blind > small_blind),
    pot_size          INTEGER NOT NULL CHECK (pot_size >= small_blind + big_blind),
    winner_player_id  INTEGER NOT NULL
                              REFERENCES players (player_id)
                              ON UPDATE CASCADE ON DELETE RESTRICT,
    final_street_id   INTEGER NOT NULL
                              REFERENCES streets (street_id),
    went_to_showdown  INTEGER NOT NULL CHECK (went_to_showdown IN (0, 1)),
    -- Шоудаун возможен только после ривера
    CHECK (went_to_showdown = 0 OR final_street_id = 4)
) STRICT;

-- Связующая таблица «игрок в раздаче» (M:N между hands и players).
-- Хранит атрибуты, зависящие от ПАРЫ (раздача, игрок): позицию, карты, результат.
CREATE TABLE hand_players (
    hand_id         INTEGER NOT NULL
                            REFERENCES hands (hand_id) ON DELETE CASCADE,
    player_id       INTEGER NOT NULL
                            REFERENCES players (player_id) ON DELETE RESTRICT,
    position_id     INTEGER NOT NULL
                            REFERENCES positions (position_id),
    hole_cards      TEXT    NOT NULL
                            CHECK (hole_cards GLOB '[2-9TJQKA][cdhs][2-9TJQKA][cdhs]'),
    starting_stack  INTEGER NOT NULL CHECK (starting_stack > 0),
    net_result      INTEGER NOT NULL,          -- выигрыш(+)/проигрыш(-) в фишках
    PRIMARY KEY (hand_id, player_id),
    UNIQUE (hand_id, position_id),             -- одна позиция — один игрок
    CHECK (net_result >= -starting_stack)      -- нельзя проиграть больше стека
) STRICT;

-- Лог действий. amount — фишки, ДОБАВЛЕННЫЕ в банк этим действием
-- (инкремент, а не «рейз до»), поэтому SUM(amount) по раздаче = pot_size.
CREATE TABLE actions (
    action_id       INTEGER PRIMARY KEY,
    hand_id         INTEGER NOT NULL,
    player_id       INTEGER NOT NULL,
    street_id       INTEGER NOT NULL REFERENCES streets (street_id),
    action_type_id  INTEGER NOT NULL REFERENCES action_types (action_type_id),
    action_seq      INTEGER NOT NULL CHECK (action_seq >= 1),   -- порядок внутри раздачи
    amount          INTEGER NOT NULL DEFAULT 0 CHECK (amount >= 0),
    is_all_in       INTEGER NOT NULL DEFAULT 0 CHECK (is_all_in IN (0, 1)),
    UNIQUE (hand_id, action_seq),
    -- Составной FK: действовать может только игрок, сидящий в этой раздаче
    FOREIGN KEY (hand_id, player_id)
        REFERENCES hand_players (hand_id, player_id) ON DELETE CASCADE
) STRICT;


-- =============================================================================
--  3. БИЗНЕС-ПРАВИЛА (триггеры)
-- =============================================================================

-- fold/check обязаны иметь amount = 0, остальные действия — amount > 0.
CREATE TRIGGER trg_actions_amount_matches_type
BEFORE INSERT ON actions
FOR EACH ROW
WHEN (SELECT requires_amount FROM action_types
      WHERE action_type_id = NEW.action_type_id) <> (NEW.amount > 0)
BEGIN
    SELECT RAISE(ABORT, 'actions.amount is inconsistent with action type');
END;


-- =============================================================================
--  4. ИНДЕКСЫ под аналитические запросы
--     (UNIQUE(hand_id, action_seq) уже индексирует actions по hand_id)
-- =============================================================================

CREATE INDEX idx_actions_player_street  ON actions (player_id, street_id);
CREATE INDEX idx_actions_type           ON actions (action_type_id);
CREATE INDEX idx_hand_players_player    ON hand_players (player_id);
CREATE INDEX idx_hands_winner           ON hands (winner_player_id);
CREATE INDEX idx_hands_played_at        ON hands (played_at);


-- =============================================================================
--  5. ПРЕДСТАВЛЕНИЯ
-- =============================================================================

-- Денормализованный лог для удобного чтения аналитиком / выгрузки в pandas.
CREATE VIEW v_action_log AS
SELECT
    a.action_id,
    h.hand_id,
    h.played_at,
    a.action_seq,
    s.street_name,
    p.player_name,
    pos.position_code,
    hp.hole_cards,
    t.action_name,
    a.amount,
    ROUND(CAST(a.amount AS REAL) / h.big_blind, 2) AS amount_bb,
    a.is_all_in,
    h.pot_size
FROM actions            AS a
JOIN hands              AS h   ON h.hand_id = a.hand_id
JOIN hand_players       AS hp  ON hp.hand_id = a.hand_id AND hp.player_id = a.player_id
JOIN players            AS p   ON p.player_id = a.player_id
JOIN positions          AS pos ON pos.position_id = hp.position_id
JOIN streets            AS s   ON s.street_id = a.street_id
JOIN action_types       AS t   ON t.action_type_id = a.action_type_id;
