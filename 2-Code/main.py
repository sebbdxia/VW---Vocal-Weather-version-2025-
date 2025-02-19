import os
import datetime
import logging
import spacy
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

# Configuration Azure via les variables d'environnement
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "54124b94ae904eeea1d8a652a4c3d88d")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "francecentral")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "a22a068e82f080385e390ae87d801800")
WEATHER_API_URL = os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5/forecast")

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Chargement du modèle spaCy pour le français
nlp = spacy.load("fr_core_news_sm")

# Stockage en mémoire pour l'analyse et le monitoring (simulation de la base Azure)
logs = []

class WeatherResponse(BaseModel):
    location: str
    forecast: dict
    message: str = None
    transcription: str = None

@app.post("/process_command", response_model=WeatherResponse)
async def process_command(
    file: UploadFile = File(None),
    transcription: str = Form(None),
    city: str = Form(None),
    horizon: str = Form(None)
):
    try:
        if file is not None:
            audio_bytes = await file.read()
            # Appel à Azure Speech-to-Text (à implémenter réellement) pour obtenir la transcription
            transcription_result = azure_speech_to_text(audio_bytes)
            transcription_final = transcription_result
        else:
            transcription_final = transcription or ""
        
        # Extraction automatique des entités via spaCy si une transcription est disponible
        if transcription_final:
            extracted_city, extracted_horizon = spacy_analyze(transcription_final)
        else:
            extracted_city, extracted_horizon = (None, None)
        
        # Les saisies manuelles ont la priorité sur l'extraction automatique
        final_city = city if city else (extracted_city if extracted_city else "Paris")
        final_horizon = horizon if horizon else (extracted_horizon if extracted_horizon else "demain")
        
        # Récupération de la prévision météo pour la ville sélectionnée
        forecast = get_weather_forecast(final_city)
        logging.info(f"Prévision météo récupérée pour {final_city} : {forecast}")
        
        # Stockage de la commande et de la prévision pour l'analyse et le monitoring
        store_forecast_in_db(transcription_final, final_city, final_horizon, forecast)
        
        return WeatherResponse(location=final_city, forecast=forecast, 
                               message="Prévision obtenue avec succès.",
                               transcription=transcription_final)
    except Exception as e:
        logging.error("Erreur lors du traitement de la commande", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la commande")

@app.get("/analysis")
def analysis():
    total_requests = len(logs)
    # Pour l'exemple, nous renvoyons simplement le nombre total de requêtes et l'ensemble des logs stockés
    analysis_data = {
        "total_requests": total_requests,
        "logs": logs
    }
    return analysis_data

def azure_speech_to_text(audio_bytes: bytes) -> str:
    """
    Cette fonction doit envoyer l’audio à Azure Speech-to-Text en utilisant SPEECH_KEY et SPEECH_REGION.
    Ici, une transcription fictive est renvoyée pour la démonstration.
    """
    transcription = "transcrire l'audio: Paris demain"  # Exemple de transcription
    return transcription

from typing import Tuple

def spacy_analyze(text: str) -> Tuple[str, str]:
    """
    Exploite spaCy pour extraire les entités du texte.
    On recherche la ville (entité de type GPE ou LOC) et l'horizon temporel (entité de type DATE ou TIME).
    """
    doc = nlp(text)
    location = None
    horizon = None
    for ent in doc.ents:
        if ent.label_ in ["LOC", "GPE"] and not location:
            location = ent.text
        elif ent.label_ in ["DATE", "TIME"] and not horizon:
            horizon = ent.text
    if not location:
        location = "Paris"
    if not horizon:
        horizon = "demain"
    return location, horizon

def get_weather_forecast(location: str) -> dict:
    """
    Interroge l'API météo OpenWeatherMap (forecast) pour récupérer la prévision.
    """
    params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric", "lang": "fr"}
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code != 200:
        raise Exception("Erreur lors de l'appel à l'API météo")
    return response.json()

def store_forecast_in_db(transcription: str, location: str, horizon: str, forecast: dict):
    """
    Stocke les données de la commande dans la variable globale 'logs' afin de les utiliser dans l'analyse.
    """
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transcription": transcription,
        "city": location,
        "horizon": horizon,
        "forecast": forecast
    }
    logs.append(entry)
    logging.info(f"Log stocké: {entry}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
