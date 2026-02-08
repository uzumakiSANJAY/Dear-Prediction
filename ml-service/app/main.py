import os
import logging
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg2
import psycopg2.extras

from app.models.predictor import train_model, predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lottery ML Service")


def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        user=os.environ.get("POSTGRES_USER", "lottery_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "lottery_pass_2024"),
        database=os.environ.get("POSTGRES_DB", "lottery_db"),
    )


class PredictRequest(BaseModel):
    time_slot: str


class TrainRequest(BaseModel):
    time_slot: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ml-service"}


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    if request.time_slot not in ("1pm", "6pm", "8pm"):
        raise HTTPException(status_code=400, detail="Invalid time slot. Use 1pm, 6pm, or 8pm")

    result = predict(request.time_slot)
    return result


@app.post("/train")
def train_endpoint(request: TrainRequest = TrainRequest()):
    slots = [request.time_slot] if request.time_slot else ["1pm", "6pm", "8pm"]
    results = []

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        for slot in slots:
            cur.execute(
                "SELECT * FROM draw_results WHERE time_slot = %s ORDER BY draw_date ASC",
                (slot,),
            )
            rows = cur.fetchall()

            if not rows:
                results.append({
                    "status": "skipped",
                    "time_slot": slot,
                    "reason": "no data",
                })
                continue

            # Convert rows to list of dicts
            draw_data = []
            for row in rows:
                draw_data.append({
                    "draw_no": row["draw_no"],
                    "draw_date": str(row["draw_date"]),
                    "time_slot": row["time_slot"],
                    "prizes": row["prizes"],
                })

            result = train_model(slot, draw_data)
            results.append(result)

        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"results": results}
