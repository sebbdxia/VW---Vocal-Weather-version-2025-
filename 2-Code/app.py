# app.py
import time
import logging
import threading
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Request
from pydantic import BaseModel
import json
import datetime

from db import store_forecast, get_all_forecasts, get_top_cities
from speech import azure_speech_to_text
from weather import get_weather_forecast
from nlp_utils import spacy_analyze
from metrics import init_metrics

registry, REQUEST_COUNT, REQUEST_LATENCY, FORECAST_REQUESTS, ERRORS_COUNT, FEEDBACK_COUNT = init_metrics()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
user_feedbacks = []

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_time = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(elapsed_time)
    return response

def measure_latency(func):
    from functools import wraps
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        logging.info(f"Temps d'exécution de {func.__name__} : {time.time() - start:.2f} secondes")
        return result
    return wrapper

class WeatherResponse(BaseModel):
    location: str
    forecast: dict
    forecast_days: int
    message: str = None
    transcription: str = None
    mode: str = None

@app.post("/process_command", response_model=WeatherResponse)
@measure_latency
async def process_command(
    file: UploadFile = File(None),
    transcription: str = Form(None),
    city: str = Form(None),
    forecast_days: str = Form(None)
):
    try:
        if file is not None:
            audio_bytes = await file.read()
            transcription_text = azure_speech_to_text(audio_bytes)
        else:
            transcription_text = transcription or ""
        extracted_city, extracted_days = spacy_analyze(transcription_text)
        final_city = city if city else extracted_city
        if (file is not None) or (transcription_text and not forecast_days):
            final_forecast_days = extracted_days
        else:
            try:
                final_forecast_days = int(forecast_days) if forecast_days is not None else 7
            except ValueError:
                final_forecast_days = 7
        hourly_dataframe = get_weather_forecast(final_city)
        mode_used = "vocale" if file is not None or (transcription_text and forecast_days is None) else "manuel"
        store_forecast(transcription_text, final_city, final_forecast_days, hourly_dataframe.to_dict(orient="records"), mode_used)
        FORECAST_REQUESTS.inc()
        forecast_data = hourly_dataframe.to_dict(orient="records")
        return WeatherResponse(
            location=final_city,
            forecast={"hourly": forecast_data},
            forecast_days=final_forecast_days,
            message="Prévision obtenue avec succès.",
            transcription=transcription_text,
            mode=mode_used
        )
    except Exception as e:
        ERRORS_COUNT.inc()
        logging.error("Erreur lors du traitement de la commande", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la commande")

@app.get("/analysis")
def analysis():
    logs_db = get_all_forecasts()
    return {"total_requests": len(logs_db), "logs": logs_db, "feedbacks": user_feedbacks}

@app.get("/metrics")
def metrics():
    try:
        from prometheus_client import generate_latest
        data = generate_latest(registry)
        return Response(content=data, media_type="text/plain")
    except Exception as e:
        logging.error("Erreur lors de l'exposition des métriques", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de l'exposition des métriques")

@app.get("/top_cities")
def top_cities():
    return get_top_cities()

@app.get("/feedbacks")
def get_feedbacks():
    return user_feedbacks

@app.post("/feedback")
def feedback(rating: int = Form(...), comment: str = Form("")):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rating": rating,
        "comment": comment
    }
    user_feedbacks.append(entry)
    FEEDBACK_COUNT.inc()
    try:
        import psycopg2
        from config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ,
                rating INTEGER,
                comment TEXT
            );
        """)
        cur.execute("""
            INSERT INTO feedback (timestamp, rating, comment)
            VALUES (%s, %s, %s)
        """, (entry["timestamp"], entry["rating"], entry["comment"]))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du stockage du feedback PostgreSQL: {e}")
    return {"status": "Feedback enregistré"}

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_backend()
