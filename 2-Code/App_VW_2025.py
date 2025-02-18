import os
import json
import re
import requests
import psycopg2
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import azure.cognitiveservices.speech as speechsdk

# Import des librairies d'interface uniquement en mode UI
if os.getenv("MODE", "API") == "UI":
    import streamlit as st
    import pandas as pd
    import folium
    from streamlit_folium import st_folium

# Configuration des services externes
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "YOUR_AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION", "YOUR_AZURE_REGION")

# Configuration (éventuelle) d'Azure LUIS pour l'analyse NLP
LUIS_ENDPOINT = os.getenv("LUIS_ENDPOINT", "YOUR_LUIS_ENDPOINT")
LUIS_APP_ID = os.getenv("LUIS_APP_ID", "YOUR_LUIS_APP_ID")
LUIS_API_KEY = os.getenv("LUIS_API_KEY", "YOUR_LUIS_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")

# Configuration PostgreSQL (Azure)
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Initialisation de l'application FastAPI avec authentification de base
app = FastAPI()
security = HTTPBasic()

# Modèles de requête via Pydantic
class WeatherRequest(BaseModel):
    location: str
    days: int

class VoiceRequest(BaseModel):
    text: str

# Fonction d'initialisation de la base de données : création de la table si nécessaire
def initialize_database():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_queries (
                id SERIAL PRIMARY KEY,
                location TEXT,
                days INTEGER,
                response JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Erreur lors de l'initialisation de la base de données : {e}")
    finally:
        conn.close()

initialize_database()

# Fonction de reconnaissance vocale utilisant Azure Speech-to-Text
def recognize_speech():
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Veuillez parler...")
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return ""

# Fonction d'extraction d'entités à partir du texte reconnu.
# Il s'agit ici d'un stub simulant l'appel à Azure LUIS. En production, cette fonction devra appeler le service LUIS.
def extract_entities(text: str):
    # Exemple d'extraction simple via une expression régulière :
    match_location = re.search(r"à\s+([A-Za-zéèêëàâäôöûüîïç\s]+)", text, re.IGNORECASE)
    match_days = re.search(r"(\d+)\s*(jours?)", text, re.IGNORECASE)
    extracted_location = match_location.group(1).strip() if match_location else "Paris"
    extracted_days = int(match_days.group(1)) if match_days else 3
    return {"location": extracted_location, "days": extracted_days}

# Fonction pour interroger l'API météo d'OpenWeatherMap
def get_weather(location: str, days: int):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&cnt={days}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Erreur lors de l'appel à l'API météo")
    return response.json()

# Fonction de stockage des requêtes et des résultats dans la base de données
def store_query(location: str, days: int, response_data: dict):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO weather_queries (location, days, response) VALUES (%s, %s, %s)",
            (location, days, json.dumps(response_data))
        )
        conn.commit()
    except Exception as e:
        print(f"Erreur lors du stockage de la requête : {e}")
    finally:
        conn.close()

# Fonction pour récupérer l'historique des requêtes stockées
def get_stored_queries():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT id, location, days, response, timestamp FROM weather_queries ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        print(f"Erreur lors de la récupération des requêtes : {e}")
        return []
    finally:
        conn.close()

# Endpoint FastAPI recevant une requête directe avec lieu et nombre de jours
@app.post("/weather/")
def fetch_weather(request: WeatherRequest, credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "password":
        raise HTTPException(status_code=401, detail="Authentification requise")
    weather_data = get_weather(request.location, request.days)
    store_query(request.location, request.days, weather_data)
    return weather_data

# Endpoint FastAPI traitant une commande vocale (texte reconnu) pour extraire les entités
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

# Partie interface utilisateur avec Streamlit (se lance si MODE=UI)
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

# Génération d'un Dockerfile pour containeriser l'application
dockerfile_content = '''
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

with open("Dockerfile", "w") as f:
    f.write(dockerfile_content)
