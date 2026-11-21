from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


RANDOM_STATE = 42
UCI_ARCHIVE = (
    "https://archive.ics.uci.edu/static/public/597/"
    "productivity%2Bprediction%2Bof%2Bgarment%2Bemployees.zip"
)
MIRROR_CSV = (
    "https://raw.githubusercontent.com/tamjid-ahsan/"
    "Garment-worker-efficiency/main/data/garments_worker_productivity.csv"
)
MODEL_PATH = Path(__file__).with_name("garment_productivity_model.joblib")


def load_data() -> pd.DataFrame:
    try:
        with urlopen(UCI_ARCHIVE, timeout=45) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive:
            csv_name = next(
                name for name in archive.namelist()
                if name.endswith("garments_worker_productivity.csv")
            )
            csv_bytes = archive.read(csv_name)
    except Exception:
        with urlopen(MIRROR_CSV, timeout=45) as response:
            csv_bytes = response.read()
    return pd.read_csv(BytesIO(csv_bytes))


def main() -> None:
    df = load_data()
    data = df.copy()
    data.columns = data.columns.str.strip()
    data["department"] = data["department"].astype(str).str.strip().replace(
        {"sweing": "sewing"}
    )
    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%Y", errors="raise")
    data = data.drop_duplicates().dropna(
        subset=["actual_productivity", "date"]
    ).copy()
    data["month"] = data["date"].dt.month
    data["day_of_month"] = data["date"].dt.day

    sampled = (
        data.groupby("department", group_keys=False)
        .sample(frac=0.90, random_state=RANDOM_STATE)
        .sort_values(["date", "department", "team"])
        .reset_index(drop=True)
    )
    all_dates = np.array(sorted(sampled["date"].unique()))
    split_date = all_dates[int(len(all_dates) * 0.8)]
    train_data = sampled.loc[sampled["date"] < split_date].reset_index(drop=True)
    test_data = sampled.loc[sampled["date"] >= split_date].reset_index(drop=True)

    X_train = train_data.drop(columns=["date", "actual_productivity"])
    y_train = train_data["actual_productivity"]
    X_test = test_data.drop(columns=["date", "actual_productivity"])
    y_test = test_data["actual_productivity"]

    categorical_columns = ["quarter", "department", "day", "team"]
    numeric_columns = [
        column for column in X_train.columns
        if column not in categorical_columns
    ]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("outliers", QuantileTransformer(
                n_quantiles=100,
                output_distribution="uniform",
                random_state=RANDOM_STATE,
            )),
        ]), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ]), categorical_columns),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

    predicted = model.predict(X_test)
    print(
        f"train={len(X_train)}, test={len(X_test)}, "
        f"MAE={mean_absolute_error(y_test, predicted):.4f}, "
        f"RMSE={np.sqrt(mean_squared_error(y_test, predicted)):.4f}, "
        f"R2={r2_score(y_test, predicted):.4f}"
    )
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
