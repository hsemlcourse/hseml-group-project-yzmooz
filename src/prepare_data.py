from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    ID_COL,
    RANDOM_STATE,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    TARGET_COL,
    ensure_raw_data_exists,
)
from src.preprocessing import preprocess_features


def save_processed_splits() -> None:
    ensure_raw_data_exists()
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_TRAIN_PATH)
    kaggle_test = pd.read_csv(RAW_TEST_PATH)

    train = train.drop_duplicates()

    x = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    train_processed = prepare_part(x_train, y_train)
    val_processed = prepare_part(x_val, y_val)
    test_processed = prepare_part(x_test, y_test)
    kaggle_test_processed = prepare_kaggle_test(kaggle_test)

    train_processed.to_csv("data/processed/train_processed.csv", index=False)
    val_processed.to_csv("data/processed/val_processed.csv", index=False)
    test_processed.to_csv("data/processed/test_processed.csv", index=False)
    kaggle_test_processed.to_csv(
        "data/processed/kaggle_test_processed.csv",
        index=False,
    )

    print("Processed datasets saved:")
    print("data/processed/train_processed.csv", train_processed.shape)
    print("data/processed/val_processed.csv", val_processed.shape)
    print("data/processed/test_processed.csv", test_processed.shape)
    print("data/processed/kaggle_test_processed.csv", kaggle_test_processed.shape)


def prepare_part(x_part: pd.DataFrame, y_part: pd.Series) -> pd.DataFrame:
    order_ids = x_part[ID_COL].reset_index(drop=True)
    features = x_part.drop(columns=[ID_COL])

    processed_features = preprocess_features(features).reset_index(drop=True)

    result = processed_features.copy()
    result[ID_COL] = order_ids
    result[TARGET_COL] = y_part.reset_index(drop=True)

    return result


def prepare_kaggle_test(kaggle_test: pd.DataFrame) -> pd.DataFrame:
    order_ids = kaggle_test[ID_COL].reset_index(drop=True)
    features = kaggle_test.drop(columns=[ID_COL])

    processed_features = preprocess_features(features).reset_index(drop=True)

    result = processed_features.copy()
    result[ID_COL] = order_ids

    return result


if __name__ == "__main__":
    save_processed_splits()
