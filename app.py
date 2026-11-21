from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).with_name("garment_productivity_model.joblib")


class ProductionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(description="Дата в формате MM/DD/YYYY")
    quarter: str
    department: str
    day: str
    team: int
    targeted_productivity: float
    smv: float
    wip: float | None = None
    over_time: int
    incentive: int
    idle_time: float
    idle_men: int
    no_of_style_change: int
    no_of_workers: float


def prepare_frame(records: list[ProductionRecord]) -> pd.DataFrame:
    frame = pd.DataFrame([record.model_dump() for record in records])
    dates = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="raise")
    frame["month"] = dates.dt.month
    frame["day_of_month"] = dates.dt.day
    frame["department"] = frame["department"].str.strip().replace({"sweing": "sewing"})
    return frame.drop(columns=["date"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Garment Productivity ML Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(record: ProductionRecord):
    prediction = app.state.model.predict(prepare_frame([record]))[0]
    return {"prediction": float(prediction)}


@app.post("/predict-batch")
def predict_batch(records: list[ProductionRecord]):
    if not records:
        return {"predictions": []}
    predictions = app.state.model.predict(prepare_frame(records))
    return {"predictions": [float(value) for value in predictions]}
