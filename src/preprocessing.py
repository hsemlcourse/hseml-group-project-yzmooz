import pandas as pd


def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["invalid_product_price"] = (df["product_price"] < 0).astype(int)
    df["invalid_discount_percent"] = (
        (df["discount_percent"] < 0) | (df["discount_percent"] > 100)
    ).astype(int)
    df["invalid_product_rating"] = (
        (df["product_rating"] < 1) | (df["product_rating"] > 5)
    ).astype(int)
    df["invalid_past_return_rate"] = (
        (df["past_return_rate"] < 0) | (df["past_return_rate"] > 1)
    ).astype(int)
    df["invalid_session_length"] = (df["session_length_minutes"] < 0).astype(int)
    df["invalid_num_product_views"] = (df["num_product_views"] < 0).astype(int)

    df["product_price"] = df["product_price"].clip(lower=0)
    df["discount_percent"] = df["discount_percent"].clip(lower=0, upper=100)
    df["product_rating"] = df["product_rating"].clip(lower=1, upper=5)
    df["past_return_rate"] = df["past_return_rate"].clip(lower=0, upper=1)
    df["session_length_minutes"] = df["session_length_minutes"].clip(lower=0)
    df["num_product_views"] = df["num_product_views"].clip(lower=0)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_after_discount"] = (
        df["product_price"] * (1 - df["discount_percent"] / 100)
    )
    df["discount_amount"] = df["product_price"] * df["discount_percent"] / 100

    df["views_per_minute"] = df["num_product_views"] / (
        df["session_length_minutes"].clip(lower=1)
    )

    df["high_past_return_rate"] = (df["past_return_rate"] > 0.3).astype(int)
    df["has_discount"] = (df["discount_percent"] > 0).astype(int)

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_invalid_values(df)
    df = add_features(df)
    return df