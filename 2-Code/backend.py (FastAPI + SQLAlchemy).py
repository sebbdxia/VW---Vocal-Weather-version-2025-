import os
import uuid
import json
import datetime
import logging
import requests
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Header
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la base de données (ici SQLite pour la démo, à remplacer par Azure SQL/CosmosDB)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./weather.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle de données pour historiser les requêtes météo
class WeatherQuery(Base):
    __tablename__ = "weather_queries"
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String, unique=True, index=True)
    location = Column(String, index=True)
    time_horizon = Column(String, index=True)
    response_data = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Modèles Pydantic pour la validation des données
class WeatherRequest(BaseModel):
    location: str
    time_horizon: str

class WeatherResponse(BaseModel):
    location: str
    time_horizon: str
    forecast: dict
    message: str = "Success"

# Dépendance pour obtenir la session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentification par API Key (exemple simple)
async def verify_api_key(api_key: str = Header(...)):
    expected_api_key = os.getenv("API_KEY", "secret-api-key")
    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="API Key invalide")

# Création de l’application FastAPI
app = FastAPI(
    title="API de Prévisions Météo par Commande Vocale",
    description="Système complet intégrant transcription vocale, NLP, API météo, stockage et monitoring",
    version="1.0"
)

# Fonction simulant la transcription via Azure Speech-to-Text
def transcrire_audio(audio_bytes: bytes) -> str:
    logger.info("Transcription de l'audio...")
    # En production, appelez Azure Speech-to-Text avec les identifiants nécessaires.
    # Pour la démo, nous renvoyons une phrase fixe.
    return "Météo Paris demain"

# Fonction simulant l’extraction d’entités via Azure LUIS
def extraire_entites(texte: str) -> dict:
    logger.info("Extraction des entités par NLP...")
    # Extraction simpliste à partir du texte "Météo Paris demain"
    location = "Paris" if "Paris" in texte else "Inconnue"
    time_horizon = "demain" if "demain" in texte else "aujourd'hui"
    return {"location": location, "time_horizon": time_horizon}

# Fonction appelant une API météo externe (ici OpenWeatherMap)
def obtenir_prevision(location: str, time_horizon: str) -> dict:
    logger.info(f"Interrogation de l’API météo pour {location} ({time_horizon})...")
    api_key = os.getenv("OPENWEATHER_API_KEY", "votre_openweather_api_key")
    base_url = "http://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric",
        "lang": "fr"
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code != 200:
            logger.error("Erreur lors de l'appel à l'API météo")
            raise HTTPException(status_code=502, detail="Erreur lors de l'appel à l'API météo")
        data = response.json()
        # Pour simplifier, nous sélectionnons la première prévision.
        forecast = data.get("list", [])[0] if data.get("list") else {}
        return forecast
    except Exception as e:
        logger.exception("Exception lors de l'appel à l'API météo")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour traiter une commande vocale captée directement depuis le front-end
@app.post("/process", dependencies=[Depends(verify_api_key)])
async def process_audio(file: UploadFile = File(...), db=Depends(get_db)):
    try:
        audio_bytes = await file.read()
        # Transcription de l'audio
        transcription = transcrire_audio(audio_bytes)
        # Extraction des entités (lieu et horizon)
        entites = extraire_entites(transcription)
        location = entites.get("location")
        time_horizon = entites.get("time_horizon")
        # Obtention de la prévision météo
        forecast = obtenir_prevision(location, time_horizon)
        # Enregistrement de la requête et de la réponse en base de données
        query_record = WeatherQuery(
            query_id=str(uuid.uuid4()),
            location=location,
            time_horizon=time_horizon,
            response_data=json.dumps(forecast)
        )
        db.add(query_record)
        db.commit()
        db.refresh(query_record)
        return WeatherResponse(location=location, time_horizon=time_horizon, forecast=forecast)
    except Exception as e:
        logger.exception("Erreur lors du traitement de l'audio")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour tester directement avec une requête textuelle
@app.post("/weather", dependencies=[Depends(verify_api_key)])
def weather_by_text(request: WeatherRequest, db=Depends(get_db)):
    try:
        forecast = obtenir_prevision(request.location, request.time_horizon)
        query_record = WeatherQuery(
            query_id=str(uuid.uuid4()),
            location=request.location,
            time_horizon=request.time_horizon,
            response_data=json.dumps(forecast)
        )
        db.add(query_record)
        db.commit()
        db.refresh(query_record)
        return WeatherResponse(location=request.location, time_horizon=request.time_horizon, forecast=forecast)
    except Exception as e:
        logger.exception("Erreur lors du traitement de la requête texte")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour recevoir du feedback et améliorer la précision (feedback loop)
class Feedback(BaseModel):
    query_id: str
    actual_forecast: dict

@app.post("/feedback", dependencies=[Depends(verify_api_key)])
def feedback(feedback: Feedback, db=Depends(get_db)):
    try:
        record = db.query(WeatherQuery).filter(WeatherQuery.query_id == feedback.query_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Requête introuvable")
        # Ici, vous pourriez comparer record.response_data avec feedback.actual_forecast
        logger.info(f"Feedback reçu pour la requête {feedback.query_id}")
        return {"message": "Feedback reçu. Merci pour votre contribution."}
    except Exception as e:
        logger.exception("Erreur lors du traitement du feedback")
        raise HTTPException(status_code=500, detail=str(e))
