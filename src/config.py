from pathlib import Path

RANDOM_STATE = 42

TARGET_COL = "returned"
ID_COL = "order_id"

RAW_TRAIN_PATH = "data/raw/train.csv"
RAW_TEST_PATH = "data/raw/test.csv"
RAW_SAMPLE_SUBMISSION_PATH = "data/raw/sample_submission.csv"

CP1_EXPERIMENTS_PATH = "report/cp1_experiments.csv"
CP1_MODEL_PATH = "models/cp1_best_model.joblib"


def ensure_raw_data_exists() -> None:
    required_files = [
        RAW_TRAIN_PATH,
        RAW_TEST_PATH,
        RAW_SAMPLE_SUBMISSION_PATH,
    ]

    missing_files = [path for path in required_files if not Path(path).exists()]

    if missing_files:
        missing_list = ", ".join(missing_files)
        raise FileNotFoundError(
            "Required raw data files are missing: "
            f"{missing_list}. Download the Kaggle dataset and place "
            "train.csv, test.csv, and sample_submission.csv into data/raw/."
        )
