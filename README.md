[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project: Prediction Of Product Return

Студент: Сенчуков Егор Дмитриевич  
Группа: 237

## Описание проекта

Цель проекта: построить модель бинарной классификации, которая по данным о клиенте,
товаре и заказе предсказывает, будет ли товар возвращен покупателем.

- Задача: бинарная классификация
- Таргет: `returned`
- Основная метрика: `ROC-AUC`
- Датасет: [E-Commerce Dataset](https://www.kaggle.com/datasets/hakdevelopment/e-commerce-dataset)

Для CP1 в проекте реализованы:

- первичный EDA и описание данных;
- очистка и feature engineering;
- воспроизводимый train/val/test split;
- baseline-модель без feature engineering;
- несколько первых моделей для сравнения;
- сохранение подготовленных данных, таблицы экспериментов и лучшей модели.

## Структура репозитория

```text
.
├── data
│   ├── raw
│   │   ├── .gitkeep
│   │   └── README.md
│   ├── processed
│   │   ├── .gitkeep
│   │   └── README.md
├── models
│   ├── .gitkeep
│   └── README.md
├── notebooks
│   ├── 01_eda.ipynb
│   └── 2_baseline.ipynb
├── report
│   ├── cp1_experiments.csv
│   └── report.md
├── src
│   ├── config.py
│   ├── modeling.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   └── train_cp1.py
├── tests
│   └── test_pipeline.py
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

## Что делается в preprocessing

В [`src/preprocessing.py`](src/preprocessing.py):

- создаются флаги невалидных значений;
- числовые аномалии клипуются до допустимых диапазонов;
- добавляются новые признаки:
  `price_after_discount`, `discount_amount`, `views_per_minute`,
  `high_past_return_rate`, `has_discount`.

После preprocessing число модельных признаков увеличивается с `14` до `25`.

## Воспроизводимый запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# скачать датасет с Kaggle:
# https://www.kaggle.com/datasets/hakdevelopment/e-commerce-dataset
# и положить файлы train.csv, test.csv, sample_submission.csv в data/raw/

python src\prepare_data.py
python src\train_cp1.py
python -m pytest -q
python -m ruff check src tests
```

Что получится после запуска:

- `data/processed/*.csv` с готовыми сплитами;
- `report/cp1_experiments.csv` с метриками экспериментов;
- `models/cp1_best_model.joblib` с лучшей моделью CP1.

Если сырых CSV нет, скрипты завершатся с понятной ошибкой и подскажут, какие файлы
нужно добавить.

Во всех сплитах и моделях используется `random_state = 42`.

## Результаты CP1

Валидационные результаты:

| Модель                         | ROC-AUC | PR-AUC |     F1 | Precision | Recall | Accuracy |
|--------------------------------|--------:|-------:|-------:|----------:|-------:|---------:|
| `random_forest_depth12`        |  0.5915 | 0.5581 | 0.5454 |    0.5391 | 0.5519 |   0.5634 |
| `logistic_regression_features` |  0.5875 | 0.5557 | 0.5500 |    0.5360 | 0.5647 |   0.5614 |
| `logistic_regression_baseline` |  0.5866 | 0.5550 | 0.5499 |    0.5372 | 0.5632 |   0.5624 |
| `decision_tree_depth6`         |  0.5747 | 0.5375 | 0.5683 |    0.5199 | 0.6267 |   0.5481 |
| `dummy_most_frequent`          |  0.5000 | 0.4746 | 0.0000 |    0.0000 | 0.0000 |   0.5254 |

Лучшая модель на CP1: `random_forest_depth12`.

Результат на внутреннем test split:

- `ROC-AUC = 0.5888`
- `PR-AUC = 0.5597`
- `F1 = 0.5419`
- `Accuracy = 0.5591`

## Полезные ссылки

- Отчёт: [`report/report.md`](report/report.md)
- Таблица экспериментов: [`report/cp1_experiments.csv`](report/cp1_experiments.csv)
- EDA: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
- Baseline notebook: [`notebooks/2_baseline.ipynb`](notebooks/2_baseline.ipynb)
