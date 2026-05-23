[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)

# ML Project - Prediction Of Product Return

Студент: Сенчуков Егор Дмитриевич

Группа: 237

Чекпоинт: CP3

## Описание проекта

Цель проекта - построить модель бинарной классификации, которая по данным о
клиенте, товаре и заказе предсказывает, будет ли товар возвращён покупателем.

- Задача: бинарная классификация.
- Таргет: `returned`.
- Основная метрика: `ROC-AUC`.
- Датасет: [E-Commerce Dataset](https://www.kaggle.com/datasets/hakdevelopment/e-commerce-dataset).

В CP3 добавлен локальный деплой:

- FastAPI API для запросов к модели;
- Streamlit-интерфейс для ручного ввода признаков;
- обновлённый финальный отчёт в `report/report.md`.

## Структура репозитория

```text
.
├── data
│   ├── raw
│   └── processed
├── models
│   ├── cp2_best_model.joblib
│   └── README.md
├── notebooks
│   ├── 01_eda.ipynb
│   └── 2_baseline.ipynb
├── report
│   ├── screenshots
│   │   ├── streamlit_ui.png
│   │   └── swagger_ui.png
│   ├── cp1_experiments.csv
│   ├── cp2_experiments.csv
│   └── report.md
├── src
│   ├── api.py
│   ├── config.py
│   ├── modeling.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── streamlit_app.py
│   ├── train_cp1.py
│   └── train_cp2.py
├── tests
│   ├── conftest.py
│   ├── test_api.py
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

Сырые данные не коммитятся. Финальная модель `models/cp2_best_model.joblib`
коммитится для CP3, чтобы API и Streamlit запускались без переобучения.

## Данные

Используется `train.csv` размером `200000 x 16`:

- 14 исходных признаков для модели;
- 1 технический идентификатор `order_id`;
- 1 таргет `returned`.

Файл `test.csv` содержит `50000` строк и используется только как внешний тест Kaggle
без таргета.

`train.csv` содержит `200000` строк и `16` колонок: `14` исходных признаков,
`order_id` и таргет `returned`.

## Локальный запуск ML-пайплайна

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src\prepare_data.py
python src\train_cp2.py
python -m pytest -q
python -m ruff check src tests
```

## Запуск API

```bash
uvicorn src.api:app --reload
```

После запуска:

- healthcheck: http://localhost:8000/health
- документация Swagger: http://localhost:8000/docs

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"customer_age\":25,\"product_price\":11.67,\"discount_percent\":43.56,\"product_rating\":1.93,\"past_purchase_count\":8,\"past_return_rate\":0.33,\"delivery_delay_days\":0.0,\"session_length_minutes\":3.27,\"num_product_views\":26,\"device_type\":\"tablet\",\"product_category\":\"sports\",\"shipping_method\":\"express\",\"payment_method\":\"paypal\",\"used_coupon\":1}"
```

Ответ содержит:

- `prediction`: `0` или `1`;
- `returned_probability`: вероятность возврата;
- `threshold`: порог классификации.

## Запуск Streamlit

В отдельном терминале, когда API уже запущен:

```bash
streamlit run src/streamlit_app.py
```

Интерфейс доступен на http://localhost:8501.

## Docker

Проверить конфигурацию:

```bash
docker compose config
```

Запустить API:

```bash
docker compose up api
```

Запустить Streamlit UI:

```bash
docker compose up ui
```

Запустить тесты и линтер:

```bash
docker compose run --rm checks
```

## Результаты модели

Лучшая модель по validation `ROC-AUC`:

- `random_forest_grid_08`
- `max_depth = 12`
- `min_samples_leaf = 100`
- `n_estimators = 200`

Test-метрики:

| ROC-AUC | PR-AUC |     F1 | Precision | Recall | Accuracy |
|--------:|-------:|-------:|----------:|-------:|---------:|
|  0.5922 | 0.5615 | 0.4722 |    0.5644 | 0.4059 |   0.5694 |

Полная таблица экспериментов лежит в `report/cp2_experiments.csv`.

## Отчёт

Финальный Markdown-отчёт: `report/report.md`.

