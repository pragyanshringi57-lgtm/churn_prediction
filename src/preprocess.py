from pathlib import Path
from typing import Optional

import pandas as pd

DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def get_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / DATA_FILENAME


def load_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else get_data_path()
    return pd.read_csv(path)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned = cleaned.drop(columns=["customerID"], errors="ignore")
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")
    cleaned = cleaned.dropna(subset=["TotalCharges"])
    return cleaned


def load_and_clean_data(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    return clean_data(load_data(csv_path))


def prepare_features(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_data = pd.get_dummies(cleaned.copy(), drop_first=True)
    x = model_data.drop(columns=["Churn_Yes"])
    y = model_data["Churn_Yes"]
    return x, y
