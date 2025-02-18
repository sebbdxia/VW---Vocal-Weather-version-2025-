import os
import json
import re
import requests
import psycopg2
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import azure.cognitiveservices.speech as speechsdk
import spacy

# Chargement du modèle français spaCy
nlp = spacy.load("fr_core_news_sm")

# Importation des modules d'interface (Streamlit) si en mode UI
if os.getenv("MODE", "API") == "UI":
    import streamlit as st
    import pandas as pd
    import folium
    from streamlit_folium import st_folium

# Configuration des services externes
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "54124b94ae904eeea1d8a652a4c3d88d")
AZURE_REGION = os.getenv("AZURE_REGION", "francecentral")
# Nous n'utilisons pas Azure LUIS pour l'analyse NLP dans cette version.
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "a22a068e82f080385e390ae87d801800")

# Configuration PostgreSQL (Azure)
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Initialisation de FastAPI avec authentification HTTP Basic
app = FastAPI()
security = HTTPBasic()

# Modèles de requête via Pydantic
class WeatherRequest(BaseModel):
    location: str
    days: int

class VoiceRequest(BaseModel):
    text: str

def initialize_database():
    connection = None  # Déclaration explicite avant le try
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_queries (
                id SERIAL PRIMARY KEY,
                location TEXT,
                days INTEGER,
                response JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        cursor.close()
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la base de données : {e}")
    finally:
        if connection is not None:
            connection.close()

initialize_database()

def recognize_speech():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Veuillez parler...")
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return ""

# Extraction d'entités via spaCy (pour le lieu et le nombre de jours)
def extract_entities(text: str):
    doc = nlp(text)
    location = None
    days = None
    for ent in doc.ents:
        # Les entités de type LOC ou GPE seront considérées comme des lieux
        if ent.label_ in ("LOC", "GPE"):
            location = ent.text
        # Les entités de type CARDINAL seront utilisées pour le nombre de jours
        elif ent.label_ == "CARDINAL":
            try:
                days = int(ent.text)
            except ValueError:
                pass
    if location is None:
        location = "Paris"
    if days is None:
        days = 3
    return {"location": location, "days": days}

def get_weather(location: str, days: int):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&cnt={days}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Erreur lors de l'appel à l'API météo")
    return response.json()

def store_query(location: str, days: int, response_data: dict):
    connection = None
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO weather_queries (location, days, response) VALUES (%s, %s, %s)",
            (location, days, json.dumps(response_data))
        )
        connection.commit()
        cursor.close()
    except Exception as e:
        print(f"Erreur lors du stockage de la requête : {e}")
    finally:
        if connection is not None:
            connection.close()

def get_stored_queries():
    connection = None
    try:
        connection = psycopg2.connect(DATABASE_URL)
        cursor = connection.cursor()
        cursor.execute("SELECT id, location, days, response, timestamp FROM weather_queries ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except Exception as e:
        print(f"Erreur lors de la récupération des requêtes : {e}")
        return []
    finally:
        if connection is not None:
            connection.close()

@app.post("/weather/")
def fetch_weather(request: WeatherRequest, credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "password":
        raise HTTPException(status_code=401, detail="Authentification requise")
    weather_data = get_weather(request.location, request.days)
    store_query(request.location, request.days, weather_data)
    return weather_data

@app.post("/weather/voice/")
def fetch_weather_from_voice(request: VoiceRequest, credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "password":
        raise HTTPException(status_code=401, detail="Authentification requise")
    entities = extract_entities(request.text)
    location = entities["location"]
    days = entities["days"]
    weather_data = get_weather(location, days)
    store_query(location, days, weather_data)
    return {"location": location, "days": days, "weather": weather_data}

if os.getenv("MODE", "API") == "UI":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Prévisions Météo", "Analyse et Monitoring"])

    if page == "Prévisions Météo":
        st.title("Prévisions Météo")
        location_input = st.text_input("Entrez un lieu :", "Paris")
        days_input = st.slider("Nombre de jours", 1, 7, 3)

        # Affichage d'une carte interactive avec Folium
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=5)
        st_folium(m, width=700, height=500)

        if st.button("Obtenir la météo"):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/weather/",
                    auth=("admin", "password"),
                    json={"location": location_input, "days": days_input}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.write(data)
                else:
                    st.error("Erreur lors de la récupération des données météo")
            except Exception as e:
                st.error(f"Erreur : {e}")

        if st.button("Reconnaissance vocale"):
            recognized_text = recognize_speech()
            st.write(f"Texte reconnu : {recognized_text}")
            if recognized_text:
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/weather/voice/",
                        auth=("admin", "password"),
                        json={"text": recognized_text}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.write(result)
                    else:
                        st.error("Erreur lors du traitement de la commande vocale")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    elif page == "Analyse et Monitoring":
        st.title("Analyse et Monitoring")
        stored_queries = get_stored_queries()
        if stored_queries:
            df = pd.DataFrame(stored_queries, columns=["ID", "Lieu", "Jours", "Réponse", "Timestamp"])
            st.write(df)
        else:
            st.write("Aucune donnée enregistrée.")

    st.sidebar.text("Application Dockerisée")

dockerfile_content = '''
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)
