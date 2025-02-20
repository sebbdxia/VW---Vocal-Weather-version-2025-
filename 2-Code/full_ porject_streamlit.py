import os
import datetime
import logging
import spacy
import requests
import uvicorn
import threading
import time
import re
import tempfile
import json
from typing import Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

import streamlit as st

import requests_cache
import pandas as pd
from retry_requests import retry

# Pour le graphique interactif
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Charger les variables d'environnement (fichier .env)
from dotenv import load_dotenv
load_dotenv('.env')

# Configuration PostgreSQL (Azure)
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")

# -------------------- Setup de la session HTTP avec cache --------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# -------------------- Backend FastAPI --------------------
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")

app = FastAPI()
logging.basicConfig(level=logging.INFO)

nlp = spacy.load("fr_core_news_sm")
logs = []

class WeatherResponse(BaseModel):
    location: str
    forecast: dict
    forecast_days: int
    message: str = None
    transcription: str = None

@app.post("/process_command", response_model=WeatherResponse)
async def process_command(
    file: UploadFile = File(None),
    transcription: str = Form(None),
    city: str = Form(None),
    forecast_days: str = Form(None)  # renseigné en mode manuel
):
    try:
        # Si un fichier audio est fourni (commande vocale), on utilise Azure Speech.
        if file is not None:
            audio_bytes = await file.read()
            transcription_result = azure_speech_to_text(audio_bytes)
            transcription_final = transcription_result
        else:
            transcription_final = transcription or ""
        
        # Extraction via spaCy : récupère la ville et le nombre de jours demandés.
        extracted_city, extracted_days = spacy_analyze(transcription_final)
        logging.info(f"Extraction: Ville='{extracted_city}', Jours='{extracted_days}'")
        final_city = city if city else extracted_city
        
        # Si forecast_days n'est pas fourni et que la commande vocale a été utilisée, on prend celle extraite.
        if (file is not None) or (transcription_final and not forecast_days):
            final_forecast_days = extracted_days
        else:
            final_forecast_days = int(forecast_days) if forecast_days is not None else 7
        
        hourly_dataframe = get_weather_forecast(final_city)
        logging.info(f"Prévision météo récupérée pour {final_city}")
        store_forecast_in_db(transcription_final, final_city, final_forecast_days, hourly_dataframe)
        
        forecast_dict = hourly_dataframe.to_dict(orient="records")
        return WeatherResponse(
            location=final_city,
            forecast={"hourly": forecast_dict},
            forecast_days=final_forecast_days,
            message="Prévision obtenue avec succès.",
            transcription=transcription_final
        )
    except Exception as e:
        logging.error("Erreur lors du traitement de la commande", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la commande")

@app.get("/analysis")
def analysis():
    return {"total_requests": len(logs), "logs": logs}

def azure_speech_to_text(audio_bytes: bytes = None) -> str:
    """
    Utilise le SDK Azure Speech pour la reconnaissance vocale.
    Si audio_bytes est fourni, il est sauvegardé dans un fichier temporaire.
    """
    import azure.cognitiveservices.speech as speechsdk
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_filename = temp_audio.name
        audio_config = speechsdk.audio.AudioConfig(filename=temp_filename)
    else:
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        recognized_text = result.text
    else:
        recognized_text = ""
    
    if audio_bytes:
        os.remove(temp_filename)
    
    return recognized_text

def azure_speech_from_microphone() -> str:
    """
    Capture la voix via le microphone en utilisant Azure Speech.
    """
    import azure.cognitiveservices.speech as speechsdk
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    st.info("Veuillez parler...")
    result = speech_recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return ""

def spacy_analyze(text: str) -> Tuple[str, int]:
    """
    Extrait la ville et le nombre de jours demandés depuis la commande.
    Recherche une mention du type "sur 5 jours". Par défaut, renvoie 7 jours.
    """
    doc = nlp(text)
    location = None
    forecast_days = None
    match = re.search(r"sur\s+(\d+)\s+jours", text, re.IGNORECASE)
    if match:
        try:
            num = int(match.group(1))
            if num in [3, 5, 7]:
                forecast_days = num
        except Exception as e:
            logging.error(f"Erreur extraction jours: {e}")
    if not forecast_days:
        for token in doc:
            if token.like_num:
                try:
                    num = int(token.text)
                    if num in [3, 5, 7]:
                        forecast_days = num
                        break
                except:
                    continue
    for ent in doc.ents:
        if ent.label_ in ["LOC", "GPE"] and not location:
            location = ent.text
    if not location:
        location = "Paris"
    if not forecast_days:
        forecast_days = 7
    return location, forecast_days

def get_coordinates(city_name: str) -> Tuple[float, float]:
    geocode_url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "json"}
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(geocode_url, params=params, headers=headers)
    data = r.json()
    if not data:
        raise Exception(f"Ville introuvable : {city_name}")
    return float(data[0]["lat"]), float(data[0]["lon"])

def get_weather_forecast(city_name: str) -> pd.DataFrame:
    lat, lon = get_coordinates(city_name)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloudcover,windspeed_10m",
        "timezone": "auto"
    }
    response = retry_session.get(url, params=params)
    data = response.json()
    times = pd.to_datetime(data['hourly']['time'])
    df = pd.DataFrame({
        "date": times,
        "temperature_2m": data['hourly']['temperature_2m'],
        "cloudcover": data['hourly']['cloudcover'],
        "windspeed_10m": data['hourly']['windspeed_10m'],
        "pm2_5": [12.3] * len(times)
    })
    return df

def store_forecast_in_db(transcription: str, location: str, forecast_days: int, forecast_df: pd.DataFrame):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transcription": transcription,
        "city": location,
        "forecast_days": forecast_days,
        "forecast": forecast_df.to_dict(orient="records")
    }
    # Stockage en mémoire
    logs.append(entry)
    logging.info(f"Log stocké: {entry}")
    
    # Stockage dans PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ,
                transcription TEXT,
                city TEXT,
                forecast_days INTEGER,
                forecast JSONB
            );
        """)
        cur.execute("""
            INSERT INTO forecasts (timestamp, transcription, city, forecast_days, forecast)
            VALUES (%s, %s, %s, %s, %s)
        """, (entry["timestamp"], entry["transcription"], entry["city"], entry["forecast_days"], json.dumps(entry["forecast"])))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du stockage PostgreSQL: {e}")

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Lancer le backend en arrière-plan si nécessaire
if "backend_started" not in st.session_state:
    st.session_state.backend_started = True
    threading.Thread(target=run_backend, daemon=True).start()
    time.sleep(1)

# -------------------- Interface Streamlit --------------------
st.title("Application Météo – Commande vocale / manuelle (Open-Meteo)")

# Utiliser st.session_state pour conserver la réponse
if "forecast_response" not in st.session_state:
    st.session_state.forecast_response = None

tab1, tab2 = st.tabs(["Prévisions (Python)", "Analyse & Monitoring"])

with tab1:
    st.header("Envoyer la commande au backend")
    mode = st.radio("Choisissez votre mode de commande :", ("Enregistrement par micro", "Charger un fichier audio", "Manuelle"))
    
    audio_bytes = None
    transcription_input = ""
    city_input = ""
    forecast_days_input = None
    if mode != "Enregistrement par micro":
        forecast_days_input = st.selectbox("Nombre de jours de prévision", options=[3, 5, 7], index=2)
    
    if mode == "Enregistrement par micro":
        st.subheader("Enregistrement par micro")
        if st.button("Démarrer la transcription vocale"):
            transcription_input = azure_speech_from_microphone()
            st.write("Transcription obtenue :", transcription_input)
            # Envoi automatique de la commande vocale
            backend_url = "http://localhost:8000/process_command"
            data = {"transcription": transcription_input}
            if city_input:
                data["city"] = city_input
            response = requests.post(backend_url, data=data)
            if response.status_code == 200:
                result = response.json()
                st.session_state.forecast_response = result
                st.success(f"Prévision pour {result['location']} (commande vocale)")
            else:
                st.error(f"Erreur (code {response.status_code}) lors de la commande vocale")
        city_input = st.text_input("Ville (optionnel)")
    elif mode == "Charger un fichier audio":
        st.subheader("Charger un fichier audio")
        uploaded_file = st.file_uploader("Sélectionnez votre fichier audio", type=["wav", "mp3"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes, format="audio/wav")
        transcription_input = st.text_input("Transcription (optionnelle)")
        city_input = st.text_input("Ville (optionnel)")
    else:
        st.subheader("Commande manuelle")
        transcription_input = st.text_input("Transcription de la commande (facultatif)")
        city_input = st.text_input("Ville")
    
    if mode != "Enregistrement par micro" and st.button("Envoyer la commande"):
        backend_url = "http://localhost:8000/process_command"
        files = {}
        data = {}
        if audio_bytes is not None:
            files["file"] = ("audio.wav", audio_bytes, "audio/wav")
        if transcription_input:
            data["transcription"] = transcription_input
        if city_input:
            data["city"] = city_input
        if forecast_days_input is not None:
            data["forecast_days"] = str(forecast_days_input)
        try:
            response = requests.post(backend_url, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prévision pour {result['location']}")
                st.session_state.forecast_response = result
            else:
                st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande")
        except Exception as e:
            st.error("Impossible de joindre le backend. Vérifiez qu'il est démarré et accessible.")
    
    # Bouton pour afficher les résultats (si une commande vocale a déjà été envoyée)
    if st.session_state.forecast_response and st.button("Afficher les résultats de la commande vocale"):
        result = st.session_state.forecast_response
        st.subheader("Prévisions actuelles")
        final_days = result["forecast_days"]
        df = pd.DataFrame(result["forecast"]["hourly"])
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df_filtered = df[df['hour'] == 12].sort_values(by='date').head(final_days)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Température (°C)", "Ciel (%)", "Vent (km/h)"))
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['temperature_2m'],
                                 mode='lines+markers', marker=dict(color='red')),
                                 row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['cloudcover'],
                                 mode='lines+markers', marker=dict(color='blue')),
                                 row=2, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['windspeed_10m'],
                                 mode='lines+markers', marker=dict(color='green')),
                                 row=3, col=1)
        fig.update_layout(height=600, title_text="Prévisions de Midi sur {} jours".format(final_days), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Détails des prévisions")
        st.dataframe(df_filtered[['date', 'temperature_2m', 'cloudcover', 'windspeed_10m', 'pm2_5']].rename(
            columns={
                "date": "Date",
                "temperature_2m": "Température (°C)",
                "cloudcover": "Ciel (%)",
                "windspeed_10m": "Vent (km/h)",
                "pm2_5": "Pollution (µg/m³)"
            }
        ))
        if result.get("transcription"):
            st.write("Transcription utilisée :", result["transcription"])
    
with tab2:
    st.header("Analyse et Monitoring")
    try:
        analysis_url = "http://localhost:8000/analysis"
        response = requests.get(analysis_url)
        if response.status_code == 200:
            data_analysis = response.json()
            st.write("Nombre total de requêtes :", data_analysis.get("total_requests", 0))
            st.write("Logs récents :")
            st.json(data_analysis.get("logs", []))
        else:
            st.error("Erreur lors de la récupération des données d'analyse.")
    except Exception as e:
        st.error("Impossible de joindre le backend pour l'analyse.")
