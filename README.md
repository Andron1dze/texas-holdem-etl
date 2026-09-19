# ♠️ Texas Hold'em ETL & Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-vectorized-013243?logo=numpy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-STRICT%20tables-003B57?logo=sqlite&logoColor=white)

Мини-ETL пайплайн для анализа раздач No-Limit Texas Hold'em (6-max):
генерация логически связного лога → нормализация → валидация → загрузка в SQLite → SQL-аналитика и визуализация.

> Проект демонстрирует навыки **Data Analyst / Data Engineer**: проектирование нормализованной схемы,
> ETL на pandas/NumPy, контроль качества данных, аналитические SQL-запросы.

---

## 📌 Содержание
- [Архитектура пайплайна](#-архитектура-пайплайна)
- [Схема базы данных](#-схема-базы-данных)
- [Модель генерации данных](#-модель-генерации-данных)
- [Быстрый старт](#-быстрый-старт)
- [Структура репозитория](#-структура-репозитория)
- [Контроль качества данных](#-контроль-качества-данных)
- [Roadmap](#-roadmap)

---

## 🏗 Архитектура пайплайна

```mermaid
flowchart LR
    A[EXTRACT<br/>симуляция 1000 раздач<br/>NumPy + pandas] --> B[(data/raw/<br/>hand_log.csv)]
    B --> C[TRANSFORM<br/>нормализация,<br/>суррогатные ключи]
    C --> D[VALIDATE<br/>7 бизнес-инвариантов]
    D --> E[LOAD<br/>pandas.to_sql]
    E --> F[(poker_analytics.db<br/>SQLite)]
    F --> G[SQL-аналитика<br/>+ Matplotlib]
```

## 🗄 Схема базы данных

```mermaid
erDiagram
    players      ||--o{ hand_players : "сидит в"
    hands        ||--|{ hand_players : "содержит"
    hand_players ||--o{ actions      : "совершает"
    players      ||--o{ hands        : "выигрывает"
    positions    ||--o{ hand_players : ""
    streets      ||--o{ actions      : ""
    streets      ||--o{ hands        : "final_street"
    action_types ||--o{ actions      : ""

    players      { INTEGER player_id PK
                   TEXT player_name UK }
    hands        { INTEGER hand_id PK
                   TEXT played_at
                   INTEGER pot_size
                   INTEGER winner_player_id FK
                   INTEGER final_street_id FK
                   INTEGER went_to_showdown }
    hand_players { INTEGER hand_id PK,FK
                   INTEGER player_id PK,FK
                   INTEGER position_id FK
                   TEXT hole_cards
                   INTEGER net_result }
    actions      { INTEGER action_id PK
                   INTEGER hand_id FK
                   INTEGER player_id FK
                   INTEGER street_id FK
                   INTEGER action_type_id FK
                   INTEGER action_seq
                   INTEGER amount }
```

Ключевые решения:
- **STRICT-таблицы** — SQLite отклоняет значения неверного типа.
- **Суммы в фишках (INTEGER)**, а не REAL — никаких ошибок округления.
- **Справочники** `positions`, `streets`, `action_types` с флагами `is_voluntary` / `is_aggressive` — метрики VPIP, PFR, AF считаются одним `JOIN`.
- **Составной FK** `actions(hand_id, player_id) → hand_players` — действовать может только игрок, сидящий в раздаче.
- **Триггер** запрещает `fold`/`check` с ненулевой суммой и `call`/`bet` без суммы.

## 🎲 Модель генерации данных

| Компонент | Реализация |
|---|---|
| Игроки | 30 игроков 4 архетипов: **Nit, TAG, LAG, Fish** со скрытыми параметрами стиля |
| Раздача карт | Векторная перетасовка колоды для всех раздач сразу (`argsort` случайной матрицы) |
| Сила руки | Формула Чена → перцентиль среди всех 1326 комбинаций |
| Префлоп | Диапазоны по позициям (UTG ⟶ BTN), опен 2.5 BB, 3-бет x3/x4, 4-бет x2.3, лимпы |
| Постфлоп | Вэлью-ставки, блефы, рейзы, коллы по шансам банка (pot odds) |
| Шоудаун | Побеждает лучшая «сила руки» на ривере |

Скрытые параметры игроков сохраняются в `data/raw/player_profiles.csv` — это ground truth,
по которому можно проверить, восстанавливает ли аналитика стиль игры (VPIP / PFR / AF).

## 🚀 Быстрый старт

```bash
git clone https://github.com/Andron1dze/texas-holdem-etl.git
cd texas-holdem-etl
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data_loader.py                    # 1000 раздач, seed=42
python src/data_loader.py --hands 5000 --seed 7
```

Пример лога запуска:
```
Extract: 1000 hands, ~10k actions, 30 players
Validate: 7/7 checks passed
Load: players 30 | hands 1000 | hand_players 6000 | actions ~10k
Verify: row counts match, FK and integrity checks passed
```

## 📁 Структура репозитория

```
texas-holdem-etl/
├── README.md
├── requirements.txt
├── .gitignore
├── sql/
│   ├── database_setup.sql      # DDL: таблицы, справочники, триггер, индексы, view
│   └── analytics_queries.sql   # (этап 2) VPIP/PFR/AF, винрейт по позициям
├── src/
│   ├── data_loader.py          # ETL: generate → transform → validate → load
│   └── analytics.py            # (этап 2) SQL → pandas → Matplotlib
├── data/
│   ├── raw/                    # сырой слой: hand_log.csv, player_profiles.csv
│   └── poker_analytics.db      # генерируется, в .gitignore
├── notebooks/                  # (этап 3) EDA и отчёт
├── reports/figures/            # графики для README
└── tests/                      # (этап 2) pytest на инварианты
```

## ✅ Контроль качества данных

| Уровень | Проверки |
|---|---|
| Python (до загрузки) | zero-sum по каждой раздаче · банк = сумма ставок · 6 игроков за столом · победитель сидит в раздаче · никто не проигрывает больше стека · сумма соответствует типу действия · уникальность порядка действий |
| SQLite (при загрузке) | STRICT-типы · PK / FK / UNIQUE / CHECK · триггер на суммы |
| SQLite (после загрузки) | `PRAGMA foreign_key_check` · `PRAGMA integrity_check` · сверка количества строк |

## 🗺 Roadmap
- [x] Этап 1 — схема БД и ETL-загрузчик
- [ ] Этап 2 — аналитические SQL-запросы: VPIP, PFR, 3-bet %, AF, WTSD, bb/100 по позициям
- [ ] Этап 3 — визуализация (Matplotlib): профили игроков, винрейт, распределение банков
- [ ] Этап 4 — тесты (pytest) и CI (GitHub Actions)

## 👤 Автор
**Andron1dze** — [GitHub](https://github.com/Andron1dze)
