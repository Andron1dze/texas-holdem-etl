-- =============================================================================
--  Texas Hold'em ETL & Analytics Pipeline
--  analytics_queries.sql — аналитический слой (витрины данных) поверх схемы
-- -----------------------------------------------------------------------------
--  Архитектура — три уровня представлений (каждый читает только предыдущий):
--
--    actions ──► v_preflop_action_context   контекст каждого префлоп-действия
--                    (оконная функция: сколько рейзов было ДО этого действия)
--            ──► v_hand_player_facts        факты уровня «игрок в раздаче»
--                    (флаги VPIP / PFR / 3-bet, постфлоп-агрессия, профит в BB)
--            ──► v_player_stats             ВИТРИНА: метрики по игрокам
--            ──► v_position_winrate         ВИТРИНА: винрейт по позициям
--
--  Почему так:
--    * Каждое действие читается ровно один раз: агрегаты считаются в CTE
--      и соединяются по ключу (hand_id, player_id), без коррелированных
--      подзапросов на каждую строку.
--    * Флаги считаются на уровне раздачи, а не действия: игрок, который
--      сначала заколлировал, а потом зарейзил, даёт ОДНУ раздачу в VPIP.
--    * Соединения идут по индексам: UNIQUE(hand_id, action_seq),
--      PK hand_players(hand_id, player_id), idx_actions_player_street.
--
--  Определения метрик:
--    VPIP  = раздачи с добровольным вложением префлоп (call/raise/3-bet/4-bet)
--            / все раздачи игрока. Блайнд и чек из BB — не добровольны.
--    PFR   = раздачи с префлоп-рейзом (raise/3-bet/4-bet) / все раздачи.
--    3-bet = 3-беты / возможности 3-бета. Возможность — игрок ходит
--            префлоп, когда перед ним сделан ровно один рейз (опен).
--    AF    = (bet + raise) / call на постфлопе. NULL, если коллов не было.
--    WTSD  = доля раздач с шоудауном среди раздач, где игрок увидел флоп.
--    bb/100 = суммарный профит в больших блайндах × 100 / число раздач.
--
--  Запуск: sqlite3 data/poker_analytics.db < sql/analytics_queries.sql
--  (скрипт идемпотентен: представления пересоздаются)
-- =============================================================================

DROP VIEW IF EXISTS v_position_winrate;
DROP VIEW IF EXISTS v_player_stats;
DROP VIEW IF EXISTS v_hand_player_facts;
DROP VIEW IF EXISTS v_preflop_action_context;


-- =============================================================================
--  Уровень 1. Контекст префлоп-действий
-- =============================================================================
-- raises_before — сколько агрессивных действий (raise / 3-bet / 4-bet) было
-- в раздаче СТРОГО ДО текущего. Рамка окна «до предыдущей строки»
-- исключает само действие из подсчёта.
--   raises_before = 0 → банк не открыт (можно залимпить / открыться)
--   raises_before = 1 → игрок стоит против опен-рейза → возможность 3-бета
--   raises_before = 2 → против 3-бета → возможность 4-бета
CREATE VIEW v_preflop_action_context AS
SELECT
    a.hand_id,
    a.player_id,
    a.action_seq,
    t.action_name,
    t.is_voluntary,
    t.is_aggressive,
    COALESCE(
        SUM(t.is_aggressive) OVER (
            PARTITION BY a.hand_id
            ORDER BY a.action_seq
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ),
        0
    ) AS raises_before
FROM actions      AS a
JOIN action_types AS t ON t.action_type_id = a.action_type_id
JOIN streets      AS s ON s.street_id = a.street_id
WHERE s.street_name = 'preflop';


-- =============================================================================
--  Уровень 2. Факты «игрок в раздаче» (одна строка = hand_id × player_id)
-- =============================================================================
CREATE VIEW v_hand_player_facts AS
WITH
preflop AS (
    -- MAX по булевым флагам = «хотя бы одно действие в раздаче удовлетворяет условию»
    SELECT
        hand_id,
        player_id,
        MAX(is_voluntary)            AS vpip_flag,
        MAX(is_aggressive)           AS pfr_flag,
        MAX(raises_before = 1)       AS three_bet_opportunity,
        MAX(action_name = '3-bet')   AS three_bet_flag,
        MAX(action_name = 'fold')    AS folded_preflop
    FROM v_preflop_action_context
    GROUP BY hand_id, player_id
),
postflop AS (
    SELECT
        a.hand_id,
        a.player_id,
        SUM(t.is_aggressive)         AS aggressive_actions,   -- bet + raise
        SUM(t.action_name = 'call')  AS calls,
        MAX(t.action_name = 'fold')  AS folded_postflop
    FROM actions      AS a
    JOIN action_types AS t ON t.action_type_id = a.action_type_id
    JOIN streets      AS s ON s.street_id = a.street_id
    WHERE s.street_name <> 'preflop'
    GROUP BY a.hand_id, a.player_id
)
SELECT
    hp.hand_id,
    hp.player_id,
    hp.position_id,
    CAST(hp.net_result AS REAL) / h.big_blind                   AS net_bb,
    COALESCE(pf.vpip_flag, 0)                                   AS vpip_flag,
    COALESCE(pf.pfr_flag, 0)                                    AS pfr_flag,
    COALESCE(pf.three_bet_opportunity, 0)                       AS three_bet_opportunity,
    COALESCE(pf.three_bet_flag, 0)                              AS three_bet_flag,
    -- Флоп увидел тот, кто не сбросил префлоп, если раздача дошла до флопа
    -- (street_id задаёт хронологию улиц: 1 = preflop)
    (h.final_street_id > 1 AND COALESCE(pf.folded_preflop, 0) = 0)
                                                                AS saw_flop,
    -- До шоудауна дошёл тот, кто не сбросил ни на одной улице
    (h.went_to_showdown = 1
        AND COALESCE(pf.folded_preflop, 0) = 0
        AND COALESCE(po.folded_postflop, 0) = 0)                AS went_to_showdown,
    COALESCE(po.aggressive_actions, 0)                          AS postflop_aggressive,
    COALESCE(po.calls, 0)                                       AS postflop_calls
FROM hand_players AS hp
JOIN hands        AS h  ON h.hand_id = hp.hand_id
LEFT JOIN preflop  AS pf ON pf.hand_id = hp.hand_id AND pf.player_id = hp.player_id
LEFT JOIN postflop AS po ON po.hand_id = hp.hand_id AND po.player_id = hp.player_id;


-- =============================================================================
--  Уровень 3а. ВИТРИНА: профиль игрока
-- =============================================================================
CREATE VIEW v_player_stats AS
WITH
totals AS (
    SELECT
        player_id,
        COUNT(*)                     AS hands,
        SUM(vpip_flag)               AS vpip_hands,
        SUM(pfr_flag)                AS pfr_hands,
        SUM(three_bet_opportunity)   AS three_bet_opportunities,
        SUM(three_bet_flag)          AS three_bets,
        SUM(postflop_aggressive)     AS postflop_aggressive,
        SUM(postflop_calls)          AS postflop_calls,
        SUM(saw_flop)                AS saw_flop_hands,
        SUM(went_to_showdown)        AS showdown_hands,
        SUM(net_bb)                  AS net_bb
    FROM v_hand_player_facts
    GROUP BY player_id
),
metrics AS (
    SELECT
        p.player_id,
        p.player_name,
        t.hands,
        ROUND(100.0 * t.vpip_hands / t.hands, 1)                          AS vpip_pct,
        ROUND(100.0 * t.pfr_hands  / t.hands, 1)                          AS pfr_pct,
        ROUND(100.0 * t.three_bets / NULLIF(t.three_bet_opportunities, 0), 1)
                                                                          AS three_bet_pct,
        t.three_bet_opportunities,
        ROUND(1.0 * t.postflop_aggressive / NULLIF(t.postflop_calls, 0), 2)
                                                                          AS aggression_factor,
        ROUND(100.0 * t.showdown_hands / NULLIF(t.saw_flop_hands, 0), 1)  AS wtsd_pct,
        ROUND(t.net_bb, 1)                                                AS net_bb,
        ROUND(100.0 * t.net_bb / t.hands, 2)                              AS bb_per_100
    FROM totals  AS t
    JOIN players AS p ON p.player_id = t.player_id
)
SELECT
    m.*,
    -- Эвристическая классификация стиля по двум осям:
    -- «лузовость» (VPIP) и «агрессивность» (доля PFR внутри VPIP)
    CASE
        WHEN m.vpip_pct < 14                  THEN 'Nit'
        WHEN m.pfr_pct  < 0.55 * m.vpip_pct   THEN 'Loose-Passive'
        WHEN m.vpip_pct < 24                  THEN 'TAG'
        ELSE                                       'LAG'
    END                                                    AS player_style,
    RANK()         OVER (ORDER BY m.bb_per_100 DESC)       AS winrate_rank,
    ROUND(PERCENT_RANK() OVER (ORDER BY m.vpip_pct), 2)    AS vpip_percentile
FROM metrics AS m;


-- =============================================================================
--  Уровень 3б. ВИТРИНА: винрейт по позициям (весь пул игроков)
-- =============================================================================
-- avg_net_bb и avg_sq_net_bb нужны Python-слою для стандартной ошибки:
-- SE(bb/100) = 100 · sqrt((E[x²] − E[x]²) / n) — без зависимости от
-- математических функций SQLite, которые есть не в каждой сборке.
CREATE VIEW v_position_winrate AS
SELECT
    pos.position_code,
    pos.position_name,
    pos.preflop_order,
    COUNT(*)                                        AS hands,
    ROUND(100.0 * AVG(f.vpip_flag), 1)              AS vpip_pct,
    ROUND(100.0 * AVG(f.pfr_flag), 1)               AS pfr_pct,
    ROUND(100.0 * AVG(f.net_bb), 2)                 AS bb_per_100,
    AVG(f.net_bb)                                   AS avg_net_bb,
    AVG(f.net_bb * f.net_bb)                        AS avg_sq_net_bb
FROM v_hand_player_facts AS f
JOIN positions           AS pos ON pos.position_id = f.position_id
GROUP BY pos.position_id;


-- =============================================================================
--  Итоговые выборки (витрины готовы к использованию из SQL-клиента / pandas)
-- =============================================================================

-- Профили игроков, лучшие по винрейту сверху
SELECT player_name, hands, vpip_pct, pfr_pct, three_bet_pct, aggression_factor,
       wtsd_pct, bb_per_100, player_style, winrate_rank
FROM v_player_stats
ORDER BY winrate_rank;

-- Винрейт по позициям в порядке хода префлоп
SELECT position_code, hands, vpip_pct, pfr_pct, bb_per_100
FROM v_position_winrate
ORDER BY preflop_order;
