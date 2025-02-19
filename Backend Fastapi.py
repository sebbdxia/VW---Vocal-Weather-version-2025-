from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import spacy
import psycopg2
import urllib.parse
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
import streamlit as st

app = FastAPI()

# Configuration Azure
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Vérification des variables d'environnement essentielles
if not SPEECH_KEY or not SPEECH_REGION:
    raise ValueError("Les clés Azure Speech ne sont pas configurées correctement.")

# Configuration de la base de données PostgreSQL
DB_USER = "sebastien"
DB_PASSWORD_RAW = "GRETAP4!2025***"
DB_PASSWORD = urllib.parse.quote_plus(DB_PASSWORD_RAW)
DB_HOST = "vw-sebastien.postgres.database.azure.com"
DB_NAME = "postgres"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Charger le modèle NLP de SpaCy
nlp = spacy.load("en_core_web_sm")

# Modèle de requête
class WeatherRequest(BaseModel):
    location: str
    forecast_horizon: int  # en jours

# Fonction de reconnaissance vocale
@app.post("/speech-to-text/")
def speech_to_text(audio_file: str):
    speech_config = SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = AudioConfig(filename=audio_file)
    recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = recognizer.recognize_once()
    return {"text": result.text}

# Fonction d'analyse NLP avec SpaCy
@app.post("/nlp/")
def analyze_text(text: str):
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return {"entities": entities}

# Récupération des prévisions météo
@app.post("/weather/")
def get_weather(data: WeatherRequest):
    params = {"q": data.location, "cnt": data.forecast_horizon, "appid": WEATHER_API_KEY, "units": "metric"}
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Erreur API météo")
    return response.json()

# Connexion à la base de données PostgreSQL
@app.post("/store-data/")
def store_data(data: WeatherRequest, forecast: dict):
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    query = """
    INSERT INTO WeatherData (location, forecast_horizon, forecast) VALUES (%s, %s, %s)
    """
    cursor.execute(query, (data.location, data.forecast_horizon, str(forecast)))
    conn.commit()
    cursor.close()
    conn.close()
    return {"status": "Data stored successfully"}

@app.get("/")
def home():
    return {"message": "API Météo opérationnelle"}

# Interface Streamlit
st.title("Prévisions Météo avec Reconnaissance Vocale")

st.header("1. Reconnaissance Vocale")
audio_file = st.file_uploader("Uploader un fichier audio", type=["wav", "mp3", "ogg"])
if audio_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    response = requests.post("http://localhost:8000/speech-to-text/", json={"audio_file": "temp_audio.wav"})
    if response.status_code == 200:
        text_output = response.json()["text"]
        st.write("Texte reconnu :", text_output)
    else:
        st.error("Erreur lors de la reconnaissance vocale")

st.header("2. Analyse NLP")
user_text = st.text_area("Entrer un texte pour extraire les entités :")
if st.button("Analyser"):
    response = requests.post("http://localhost:8000/nlp/", json={"text": user_text})
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Erreur dans l'analyse NLP")

st.header("3. Prévisions Météo")
location = st.text_input("Entrez une ville :")
forecast_horizon = st.slider("Horizon de prévision (en jours)", 1, 7, 3)
if st.button("Obtenir les prévisions"):
    response = requests.post("http://localhost:8000/weather/", json={"location": location, "forecast_horizon": forecast_horizon})
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Erreur lors de la récupération des prévisions météo")
