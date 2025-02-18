import urllib.parse
import logging
import tempfile
import json
import os
import requests
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, Field
import azure.cognitiveservices.speech as speechsdk

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la base de données PostgreSQL
DB_USER = "sebastien"
DB_PASSWORD_RAW = "GRETAP4!2025***"
DB_PASSWORD = urllib.parse.quote_plus(DB_PASSWORD_RAW)
DB_HOST = "vw-sebastien.postgres.database.azure.com"
DB_NAME = "postgres"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
logger.info(f"Utilisation de la DATABASE_URL: {DATABASE_URL}")

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Modèles SQLAlchemy
class WeatherRequest(Base):
    __tablename__ = "weather_requests"
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String)
    time_horizon = Column(String)
    weather_data = Column(Text)

class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String, index=True)
    actual_forecast = Column(Text)

# Ensure database schema is correctly set up
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Tables créées avec succès.")
except Exception as e:
    logger.error(f"Erreur lors de la création des tables: {e}", exc_info=True)
    raise

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Configuration WeatherAPI
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "6cc086b59b711c0461b2873acc18bc57")  # Remplacez par votre clé réelle

# Configuration Azure Speech-to-Text
AZURE_SPEECH_KEY = "54124b94ae904eeea1d8a652a4c3d88d"
AZURE_REGION = "francecentral"

# Modèle Pydantic pour le feedback
class FeedbackPayload(BaseModel):
    query_id: str = Field(..., example="12345")
    actual_forecast: dict = Field(..., example={"main": {"temp": 22.5}})

# Fonction d'extraction d'entités
def extract_entities(text: str):
    try:
        parts = text.split(',')
        location = parts[0].strip() if parts else text.strip()
        time_horizon = parts[1].strip() if len(parts) > 1 else "today"
    except Exception as e:
        logger.error("Erreur d'extraction NLP:", exc_info=True)
        location, time_horizon = text.strip(), "today"
    return location, time_horizon

app = FastAPI()

@app.post("/process")
async def process_weather(
    file: UploadFile = File(None),
    manual_location: str = Form(None),
    manual_date: str = Form(None),
    db: Session = Depends(get_db)
):
    transcription = ""
    if file is not None:
        try:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(contents)
                temp_file_path = tmp.name
        except Exception as e:
            logger.error("Erreur lors de la création du fichier temporaire:", exc_info=True)
            raise HTTPException(status_code=500, detail="Erreur lors de la création du fichier audio temporaire")
        
        try:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
            audio_config = speechsdk.AudioConfig(filename=temp_file_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcription = result.text
            else:
                logger.error("Échec de la reconnaissance vocale (Azure Speech-to-Text)")
                raise HTTPException(status_code=400, detail="Échec de la reconnaissance vocale")
        except Exception as e:
            logger.error("Erreur lors de la reconnaissance vocale:", exc_info=True)
            raise HTTPException(status_code=500, detail="Erreur lors de la reconnaissance vocale")
        finally:
            os.remove(temp_file_path)
    
    if manual_location:
        location = manual_location
    elif transcription:
        location, _ = extract_entities(transcription)
    else:
        raise HTTPException(status_code=400, detail="Aucune information pour la localité.")
    
    if manual_date:
        time_horizon = manual_date
    elif transcription:
        _, extracted_time_horizon = extract_entities(transcription)
        time_horizon = extracted_time_horizon
    else:
        time_horizon = "today"
    
    logger.info(f"Localité: {location} | Horizon: {time_horizon} | Transcription: {transcription}")
    
    # Pour WeatherAPI, si la localité est connue dans le mapping, on utilise ses coordonnées ; sinon, on utilise le nom.
    coords_mapping = {
        "paris": {"latitude": 48.8566, "longitude": 2.3522},
        "lyon": {"latitude": 45.7640, "longitude": 4.8357},
        "marseille": {"latitude": 43.2965, "longitude": 5.3698}
    }
    location_lower = location.lower()
    if location_lower in coords_mapping:
        coords = coords_mapping[location_lower]
        query = f"{coords['latitude']},{coords['longitude']}"
    else:
        query = location
    
    weather_url = "https://api.weatherapi.com/v1/forecast.json"
    params_api = {
        "key": WEATHER_API_KEY,
        "q": query,
        "days": 1,
        "lang": "fr"
    }
    try:
        response = requests.get(weather_url, params=params_api, timeout=10)
        response.raise_for_status()
        weather_json = response.json()
        logger.info("Réponse WeatherAPI: %s", str(weather_json)[:300])
    except Exception as e:
        logger.error("Erreur lors de l'appel à WeatherAPI:", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de l'appel à WeatherAPI")
    
    try:
        forecast_hours = weather_json.get("forecast", {}).get("forecastday", [{}])[0].get("hour", [])
        df_hourly = pd.DataFrame(forecast_hours)
        if not df_hourly.empty:
            df_hourly["time"] = pd.to_datetime(df_hourly["time"])
            hourly_data = {
                "date": df_hourly["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist(),
                "temperature_2m": df_hourly["temp_c"].tolist()
            }
        else:
            hourly_data = {}
    except Exception as e:
        logger.error("Erreur lors du traitement des données horaires de WeatherAPI:", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement des données horaires")
    
    weather_data = {
        "location": weather_json.get("location", {}),
        "current": weather_json.get("current", {}),
        "hourly_data": hourly_data
    }
    
    try:
        serialized_data = json.dumps(weather_data)
        logger.info("Longueur de serialized_data: %d, aperçu: %s", len(serialized_data), serialized_data[:200])
    except Exception as e:
        logger.error("Erreur lors de la sérialisation du JSON:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sérialisation du JSON: {e}")
    
    try:
        db_request = WeatherRequest(
            location=location,
            time_horizon=time_horizon,
            weather_data=serialized_data
        )
        db.add(db_request)
        db.commit()
        db.refresh(db_request)  # Refresh to get the ID and other generated fields
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur lors de l'enregistrement en base: {e} (PostgreSQL - WeatherRequest)", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement des données: {e}")
    
    return {
        "transcription": transcription,
        "location": location,
        "time_horizon": time_horizon,
        "forecast": weather_data,
        "request_id": db_request.id  # Include the request ID in the response
    }

@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload, db: Session = Depends(get_db)):
    try:
        feedback_entry = Feedback(
            query_id=payload.query_id,
            actual_forecast=json.dumps(payload.actual_forecast)
        )
        db.add(feedback_entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Erreur lors de l'enregistrement du feedback:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement du feedback: {e}")
    
    return {"message": "Feedback enregistré avec succès.", "query_id": payload.query_id}
