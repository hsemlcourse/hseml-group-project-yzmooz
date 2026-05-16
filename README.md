[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project - Prediction Of Product Return

Студент: Сенчуков Егор Дмитриевич  
Группа: 237

Чекпоинт: CP2

## Описание проекта

Цель проекта - построить модель бинарной классификации, которая по данным о
клиенте, товаре и заказе предсказывает, будет ли товар возвращён покупателем.

- Задача: бинарная классификация.
- Таргет: `returned`.
- Основная метрика: `ROC-AUC`.
- Датасет: [E-Commerce Dataset](https://www.kaggle.com/datasets/hakdevelopment/e-commerce-dataset).

В CP2 добавлены системный перебор гиперпараметров и Docker/Docker Compose для
воспроизводимого запуска.

## Структура репозитория

```text
.
├── data
│   ├── raw
│   │   ├── .gitkeep
│   │   └── README.md
│   └── processed
│       ├── .gitkeep
│       └── README.md
├── models
│   ├── .gitkeep
│   └── README.md
├── notebooks
│   ├── 01_eda.ipynb
│   └── 2_baseline.ipynb
├── report
│   ├── cp1_experiments.csv
│   ├── cp2_experiments.csv
│   └── report.md
├── src
│   ├── config.py
│   ├── modeling.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── train_cp1.py
│   └── train_cp2.py
├── tests
│   ├── conftest.py
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Данные

Используется `train.csv` размером `200000 x 16`:

- 14 исходных признаков для модели;
- 1 технический идентификатор `order_id`;
- 1 таргет `returned`.

Файл `test.csv` содержит `50000` строк и используется только как внешний тест Kaggle
без таргета.

Сырые данные не коммитятся в репозиторий, поэтому перед запуском их нужно положить в
`data/raw/`.

Ожидаемые файлы:

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

Распределение классов в `train.csv`:

- `returned = 0`: `105081` наблюдений, `52.54%`
- `returned = 1`: `94919` наблюдений, `47.46%`

В данных не обнаружено пропусков и полных дублей, но есть невалидные значения
в нескольких числовых признаках. Они обрабатываются в коде, а не игнорируются.

Код подготовки данных находится в `src/preprocessing.py` и `src/prepare_data.py`.
В preprocessing:

- создаются флаги невалидных значений `invalid_*`;
- числовые аномалии клипуются до допустимых диапазонов;
- добавляются признаки `price_after_discount`, `discount_amount`,
  `views_per_minute`, `high_past_return_rate`, `has_discount`;
- используется stratified train/validation/test split с `random_state = 42`.

После feature engineering число модельных признаков увеличивается с `14` до
`25`.

## Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# скачать датасет с Kaggle:
# https://www.kaggle.com/datasets/hakdevelopment/e-commerce-dataset
# и положить файлы train.csv, test.csv, sample_submission.csv в data/raw/

python src\prepare_data.py
python src\train_cp2.py
python -m pytest -q
python -m ruff check src tests
```

После запуска создаются:

- `data/processed/train_processed.csv`
- `data/processed/val_processed.csv`
- `data/processed/test_processed.csv`
- `data/processed/kaggle_test_processed.csv`
- `report/cp2_experiments.csv`
- `models/cp2_best_model.joblib`

## Docker

Проверить конфигурацию:

```bash
docker compose config
```

Запустить CP2-обучение:

```bash
docker compose run --rm cp2
```

Запустить тесты и линтер:

```bash
docker compose run --rm checks
```

Docker-образ строится из `python:3.11-slim`. Директории `data/`, `models/` и
`report/` подключаются как volume, поэтому локальные данные и результаты
сохраняются вне контейнера.

## Эксперименты CP2

В CP2 baseline оставлен как reference, а основные модели обучаются с grid search
по гиперпараметрам через `ParameterGrid`.

| Лучшая модель семейства        | ROC-AUC | PR-AUC |     F1 | Precision | Recall | Accuracy |
|--------------------------------|--------:|-------:|-------:|----------:|-------:|---------:|
| `random_forest_grid_08`        |  0.5922 | 0.5594 | 0.4705 |    0.5618 | 0.4047 |   0.5677 |
| `logistic_regression_grid_02`  |  0.5875 | 0.5557 | 0.5500 |    0.5360 | 0.5647 |   0.5614 |
| `extra_trees_grid_05`          |  0.5868 | 0.5535 | 0.4751 |    0.5575 | 0.4139 |   0.5659 |
| `logistic_regression_baseline` |  0.5866 | 0.5550 | 0.5499 |    0.5372 | 0.5632 |   0.5624 |
| `decision_tree_grid_04`        |  0.5755 | 0.5403 | 0.4900 |    0.5414 | 0.4475 |   0.5579 |

Полная таблица всех `39` validation-запусков и финальной test-оценки лежит в
`report/cp2_experiments.csv`.

Лучшая модель CP2 по validation `ROC-AUC`:

- `random_forest_grid_08`
- `max_depth = 12`
- `min_samples_leaf = 100`
- `n_estimators = 200`

Финальная test-оценка лучшей модели:

- `ROC-AUC = 0.5922`
- `PR-AUC = 0.5615`
- `F1 = 0.4722`
- `Precision = 0.5644`
- `Recall = 0.4059`
- `Accuracy = 0.5694`

## Полезные ссылки

- Отчёт: `report/report.md`
- CP2-таблица экспериментов: `report/cp2_experiments.csv`
- EDA: `notebooks/01_eda.ipynb`
- Baseline notebook: `notebooks/2_baseline.ipynb`
