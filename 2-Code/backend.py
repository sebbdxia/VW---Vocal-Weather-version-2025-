import urllib.parse
import logging
import json
import os
import requests

from fastapi import FastAPI, HTTPException, Depends, Form
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pydantic import BaseModel, Field

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration PostgreSQL (avec mot de passe encodé)
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

# Modèle simple pour stocker une prévision météo
class WeatherRequest(Base):
    __tablename__ = "weather_requests"
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String)
    time_horizon = Column(String)
    weather_data = Column(Text)

# Création des tables
Base.metadata.create_all(bind=engine)
logger.info("Tables créées avec succès.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

# Endpoint simplifié pour tester l'insertion en base
@app.post("/process")
async def process_weather(
    manual_location: str = Form(...),
    manual_date: str = Form(...),
    db: Session = Depends(get_db)
):
    # Pour le test, nous allons appeler WeatherAPI et insérer le JSON brut de la réponse
    weather_api_key = os.getenv("WEATHER_API_KEY", "votre_clé_weatherapi")
    # On construit la requête avec la ville manuelle (sans conversion en coordonnées)
    weather_url = "https://api.weatherapi.com/v1/forecast.json"
    params_api = {
        "key": weather_api_key,
        "q": manual_location,
        "days": 1,
        "lang": "fr"
    }
    try:
        response = requests.get(weather_url, params=params_api, timeout=10)
        response.raise_for_status()
        weather_json = response.json()
        logger.info("Réponse WeatherAPI (aperçu) : %s", str(weather_json)[:300])
    except Exception as e:
        logger.error("Erreur lors de l'appel à WeatherAPI:", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de l'appel à WeatherAPI")

    # Pour le test, on insère le JSON complet tel quel
    try:
        serialized_data = json.dumps(weather_json)
        logger.info("Longueur de serialized_data: %d", len(serialized_data))
    except Exception as e:
        logger.error("Erreur lors de la sérialisation du JSON:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sérialisation du JSON: {e}")

    # Insertion en base
    try:
        db_request = WeatherRequest(
            location=manual_location,
            time_horizon=manual_date,
            weather_data=serialized_data
        )
        db.add(db_request)
        db.commit()
        db.refresh(db_request)
    except Exception as e:
        db.rollback()
        logger.error("Erreur lors de l'enregistrement en base:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'enregistrement des données: {e}")

    return {
        "message": "Insertion réussie",
        "request_id": db_request.id,
        "weather_data": weather_json
    }
