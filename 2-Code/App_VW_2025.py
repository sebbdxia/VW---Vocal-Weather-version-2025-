import azure.cognitiveservices.speech as speechsdk
import httpx
import re
import spacy
import uvicorn
import os
import streamlit as st
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
API_WEATHER_KEY = os.getenv("API_WEATHER_KEY")

# Debug: Vérifier les variables d'environnement
print(f"AZURE_SPEECH_KEY: {AZURE_SPEECH_KEY}")
print(f"AZURE_REGION: {AZURE_REGION}")
print(f"API_WEATHER_KEY: {API_WEATHER_KEY}")

# Chargement du modèle NLP (spaCy)
nlp = spacy.load("fr_core_news_sm")

# FastAPI application
app = FastAPI()

# Reconnaissance vocale Azure
def recognize_speech():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    print("Parlez...")
    result = speech_recognizer.recognize_once()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    return None

# Extraction des intentions (lieu et date) avec NLP
def extract_location_and_date(text):
    doc = nlp(text)
    location = None
    date = "Aujourd'hui"
    
    for ent in doc.ents:
        if ent.label_ == "LOC":
            location = ent.text
        if ent.label_ in ["DATE", "TIME"]:
            date = ent.text
    
    return location or "Non spécifié", date

# Appel asynchrone à l'API météo
async def get_weather(city):
    async with httpx.AsyncClient() as client:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_WEATHER_KEY}&units=metric&lang=fr"
        response = await client.get(url)
        # Debug: Vérifier la réponse de l'API
        print(f"API Response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            data = response.json()
            return {
                "description": data["weather"][0]["description"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }
        raise HTTPException(status_code=404, detail="Ville non trouvée")

# API FastAPI
@app.get("/meteo/{ville}")
async def meteo(ville: str):
    return await get_weather(ville)

# Interface utilisateur avec Streamlit
def streamlit_interface():
    st.title("Assistant Météo Vocal")
    
    if st.button("Parlez"):
        text = recognize_speech()
        if text:
            location, date = extract_location_and_date(text)
            try:
                meteo_data = get_weather(location)
                st.write(f"Prévisions météo pour {location} ({date}) :")
                st.write(f"- {meteo_data['description']}")
                st.write(f"- Température : {meteo_data['temperature']}°C")
                st.write(f"- Humidité : {meteo_data['humidity']}%")
                st.write(f"- Vent : {meteo_data['wind_speed']} km/h")
            except HTTPException as e:
                st.error("Ville introuvable")

# Déploiement sécurisé avec Uvicorn
if __name__ == "__main__":
    import threading
    
    # Démarrer l'API en arrière-plan
    def run_api():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    api_thread = threading.Thread(target=run_api)
    api_thread.start()
    
    # Lancer l'interface utilisateur
    streamlit_interface()
