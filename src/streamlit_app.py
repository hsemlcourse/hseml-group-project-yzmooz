import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

DEVICE_TYPES = ["desktop", "mobile", "tablet"]
PRODUCT_CATEGORIES = ["beauty", "clothing", "electronics", "home", "sports", "toys"]
SHIPPING_METHODS = ["express", "same_day", "standard"]
PAYMENT_METHODS = ["apple_pay", "credit_card", "debit_card", "paypal"]

DEVICE_TYPE_LABELS = {
    "desktop": "Компьютер",
    "mobile": "Смартфон",
    "tablet": "Планшет",
}
PRODUCT_CATEGORY_LABELS = {
    "beauty": "Красота",
    "clothing": "Одежда",
    "electronics": "Электроника",
    "home": "Товары для дома",
    "sports": "Спорт",
    "toys": "Игрушки",
}
SHIPPING_METHOD_LABELS = {
    "express": "Экспресс",
    "same_day": "В день заказа",
    "standard": "Стандартная",
}
PAYMENT_METHOD_LABELS = {
    "apple_pay": "Apple Pay",
    "credit_card": "Кредитная карта",
    "debit_card": "Дебетовая карта",
    "paypal": "PayPal",
}
COUPON_LABELS = {
    1: "Да",
    0: "Нет",
}


def post_prediction(payload: dict) -> dict:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Прогноз возврата товара", layout="centered")

st.title("Прогноз возврата товара")

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        customer_age = st.number_input(
            "Возраст покупателя",
            min_value=0,
            value=25,
            step=1,
        )
        product_price = st.number_input("Цена товара", value=11.67, step=1.0)
        discount_percent = st.number_input("Размер скидки, %", value=43.56, step=1.0)
        product_rating = st.number_input("Рейтинг товара", value=1.93, step=0.1)
        past_purchase_count = st.number_input(
            "Количество прошлых покупок",
            min_value=0,
            value=8,
            step=1,
        )
        past_return_rate = st.number_input(
            "Доля прошлых возвратов",
            value=0.33,
            step=0.01,
        )
        used_coupon = st.selectbox(
            "Купон использован",
            [1, 0],
            format_func=COUPON_LABELS.get,
        )

    with right:
        delivery_delay_days = st.number_input(
            "Задержка доставки, дни",
            value=0.0,
            step=1.0,
        )
        session_length_minutes = st.number_input(
            "Длительность сессии, минуты",
            value=3.27,
            step=1.0,
        )
        num_product_views = st.number_input(
            "Количество просмотров товара",
            min_value=0,
            value=26,
            step=1,
        )
        device_type = st.selectbox(
            "Тип устройства",
            DEVICE_TYPES,
            index=2,
            format_func=DEVICE_TYPE_LABELS.get,
        )
        product_category = st.selectbox(
            "Категория товара",
            PRODUCT_CATEGORIES,
            index=4,
            format_func=PRODUCT_CATEGORY_LABELS.get,
        )
        shipping_method = st.selectbox(
            "Способ доставки",
            SHIPPING_METHODS,
            index=0,
            format_func=SHIPPING_METHOD_LABELS.get,
        )
        payment_method = st.selectbox(
            "Способ оплаты",
            PAYMENT_METHODS,
            index=3,
            format_func=PAYMENT_METHOD_LABELS.get,
        )

    submitted = st.form_submit_button("Сделать прогноз")

if submitted:
    request_payload = {
        "customer_age": int(customer_age),
        "product_price": float(product_price),
        "discount_percent": float(discount_percent),
        "product_rating": float(product_rating),
        "past_purchase_count": int(past_purchase_count),
        "past_return_rate": float(past_return_rate),
        "delivery_delay_days": float(delivery_delay_days),
        "session_length_minutes": float(session_length_minutes),
        "num_product_views": int(num_product_views),
        "device_type": device_type,
        "product_category": product_category,
        "shipping_method": shipping_method,
        "payment_method": payment_method,
        "used_coupon": int(used_coupon),
    }

    try:
        result = post_prediction(request_payload)
    except requests.RequestException as exc:
        st.error(f"Ошибка запроса к API: {exc}")
    else:
        probability = result["returned_probability"]
        prediction = result["prediction"]
        label = "Ожидается возврат" if prediction == 1 else "Возврат не ожидается"

        st.metric("Вероятность возврата", f"{probability:.3f}")
        st.write(label)
