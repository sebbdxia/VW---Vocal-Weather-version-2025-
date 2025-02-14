import os
import sys
import subprocess

# Vérification préalable pour éviter les conflits d'importation (ex: répertoire nommé "numpy")
if os.path.basename(os.getcwd()).lower() == "numpy":
    sys.exit("Erreur : Exécutez ce script depuis un répertoire différent de 'numpy' pour éviter des conflits d'importation.")

# Vérification et installation de NumPy
try:
    import numpy as np
except ImportError as e:
    print("Erreur lors de l'importation de numpy :", e)
    print("Tentative de réinstallation de numpy...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "numpy"])
        import numpy as np
        print("Numpy réinstallé avec succès, version :", np.__version__)
    except Exception as e:
        sys.exit("Impossible de réinstaller numpy automatiquement : " + str(e))

# Vérification et installation de pydantic_core (nécessaire pour FastAPI et Pydantic)
try:
    import pydantic_core
except ImportError as e:
    print("Erreur lors de l'importation de pydantic_core :", e)
    print("Tentative de réinstallation de pydantic_core...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "pydantic-core"])
        import pydantic_core
        print("pydantic_core réinstallé avec succès, version :", pydantic_core.__version__)
    except Exception as e:
        sys.exit("Impossible de réinstaller pydantic_core automatiquement : " + str(e))

# Import des autres modules nécessaires
import asyncio
import threading
import logging
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
import httpx
import spacy
import uvicorn
import streamlit as st
from fastapi import FastAPI, HTTPException

# Configuration du logging pour une traçabilité accrue
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Chargement des variables d'environnement
load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
API_WEATHER_KEY = os.getenv("API_WEATHER_KEY")

if not all([AZURE_SPEECH_KEY, AZURE_REGION, API_WEATHER_KEY]):
    logging.error("Les clés d'API nécessaires ne sont pas toutes définies dans les variables d'environnement.")
    sys.exit(1)

logging.info("Les variables d'environnement ont été chargées avec succès.")

# Chargement du modèle NLP spaCy
try:
    nlp = spacy.load("fr_core_news_sm")
    logging.info("Le modèle spaCy a été chargé avec succès.")
except Exception as e:
    logging.error("Erreur lors du chargement du modèle spaCy : %s", e)
    sys.exit(1)

# Création de l'application FastAPI
app = FastAPI()

def recognize_speech():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    logging.info("Veuillez parler...")
    result = speech_recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        logging.info("Texte reconnu : %s", result.text)
        return result.text
    logging.warning("Reconnaissance échouée, raison : %s", result.reason)
    return None

def extract_location_and_date(text):
    logging.info("Analyse du texte afin d'extraire le lieu et la date.")
    doc = nlp(text)
    location = None
    date = "Aujourd'hui"
    for ent in doc.ents:
        logging.debug("Entité détectée : %s (%s)", ent.text, ent.label_)
        if ent.label_ == "LOC":
            location = ent.text
        elif ent.label_ in ["DATE", "TIME"]:
            date = ent.text
    location = location or "Non spécifié"
    logging.info("Extraction terminée : lieu = %s, date = %s", location, date)
    return location, date

async def get_weather(city):
    logging.info("Récupération des données météo pour : %s", city)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_WEATHER_KEY}&units=metric&lang=fr"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
        except Exception as e:
            logging.error("Erreur lors de la requête API : %s", e)
            raise HTTPException(status_code=500, detail="Erreur lors de l'appel à l'API météo")
    if response.status_code == 200:
        data = response.json()
        logging.debug("Données météo reçues : %s", data)
        return {
            "description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
    logging.error("Ville non trouvée, code de réponse : %s", response.status_code)
    raise HTTPException(status_code=404, detail="Ville non trouvée")

@app.get("/meteo/{ville}")
async def meteo(ville: str):
    logging.info("Requête API reçue pour la ville : %s", ville)
    return await get_weather(ville)

async def streamlit_interface():
    st.title("Assistant Météo Vocal")
    if st.button("Parlez"):
        text = recognize_speech()
        if text:
            location, date = extract_location_and_date(text)
            try:
                meteo_data = await get_weather(location)
                st.write(f"Prévisions météo pour {location} ({date}) :")
                st.write(f"- {meteo_data['description']}")
                st.write(f"- Température : {meteo_data['temperature']}°C")
                st.write(f"- Humidité : {meteo_data['humidity']}%")
                st.write(f"- Vent : {meteo_data['wind_speed']} km/h")
            except HTTPException as e:
                st.error("Ville introuvable")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")
        else:
            st.error("Aucun texte reconnu. Veuillez réessayer.")

if __name__ == "__main__":
    # Lancement du serveur FastAPI dans un thread séparé
    def run_api():
        logging.info("Lancement du serveur FastAPI...")
        uvicorn.run(app, host="0.0.0.0", port=8000)

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    logging.info("Lancement de l'interface Streamlit...")
    asyncio.run(streamlit_interface())
