import os
import datetime
import logging
import spacy
import requests
import uvicorn
import threading
import time
import queue
import av

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

import requests_cache
import pandas as pd
from retry_requests import retry

# Pour le graphique interactif
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------- Setup de la session HTTP avec cache --------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# -------------------- Audio Recorder pour streamlit-webrtc --------------------
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = queue.Queue()
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.put(frame)
        return frame
    def get_audio_bytes(self):
        frames = []
        while not self.audio_frames.empty():
            frame = self.audio_frames.get()
            frames.append(frame.to_ndarray().tobytes())
        return b"".join(frames)

# -------------------- Backend FastAPI et configuration --------------------
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "54124b94ae904eeea1d8a652a4c3d88d")
SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "francecentral")

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Chargement du modèle spaCy pour le français
nlp = spacy.load("fr_core_news_sm")
logs = []

# On ajoute forecast_days dans la réponse
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
    forecast_days: str = Form(None)  # optionnel en mode manuel
):
    try:
        # Si un fichier audio est fourni, on tente de le transcrire
        if file is not None:
            audio_bytes = await file.read()
            transcription_result = azure_speech_to_text(audio_bytes)
            transcription_final = transcription_result
        else:
            transcription_final = transcription or ""
        
        # Extraction via spaCy : récupère la ville et le nombre de jours.
        extracted_city, extracted_days = spacy_analyze(transcription_final)
        
        # Priorité à la saisie manuelle pour la ville
        final_city = city if city else extracted_city
        
        # Détermination de la durée de prévision selon le mode
        if file is not None:
            # En commande vocale, le nombre de jours extrait est utilisé
            final_forecast_days = extracted_days
        else:
            final_forecast_days = int(forecast_days) if forecast_days is not None else 7
        
        # Récupération de la prévision météo
        hourly_dataframe = get_weather_forecast(final_city)
        logging.info(f"Prévision météo récupérée (hourly) pour {final_city}")
        
        # Stockage dans les logs
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
    total_requests = len(logs)
    return {"total_requests": total_requests, "logs": logs}

def azure_speech_to_text(audio_bytes: bytes) -> str:
    """
    Ici, la transcription est simulée.
    Pour un déploiement réel, intégrez la solution Azure Speech-to-Text.
    La transcription doit inclure la ville et idéalement une mention du nombre de jours, par exemple :
    "Prévision pour Paris sur 5 jours"
    """
    # Simulation améliorée pour le test
    return "Prévision pour Paris sur 5 jours"

from typing import Tuple
import re

def spacy_analyze(text: str) -> Tuple[str, int]:
    """
    Extrait la ville et le nombre de jours de prévision depuis la commande.
    Recherche une mention du type "sur 5 jours".
    Si aucun nombre parmi 3, 5 ou 7 n'est trouvé, renvoie 7 par défaut.
    """
    doc = nlp(text)
    location = None
    forecast_days = None
    
    # Utilisation d'une expression régulière pour détecter "sur <nombre> jours"
    match = re.search(r"sur\s+(\d+)\s+jours", text, re.IGNORECASE)
    if match:
        try:
            num = int(match.group(1))
            if num in [3, 5, 7]:
                forecast_days = num
        except:
            pass
    
    # Sinon, on parcourt les tokens pour détecter un nombre acceptable
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
                    
    # Extraction de la ville
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
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon

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
    temperature = data['hourly']['temperature_2m']
    cloudcover = data['hourly']['cloudcover']
    windspeed = data['hourly']['windspeed_10m']
    # Simulation de la pollution
    pm25 = [12.3] * len(times)
    df = pd.DataFrame({
        "date": times,
        "temperature_2m": temperature,
        "cloudcover": cloudcover,
        "windspeed_10m": windspeed,
        "pm2_5": pm25
    })
    print(df)
    return df

def store_forecast_in_db(transcription: str, location: str, forecast_days: int, forecast_df: pd.DataFrame):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transcription": transcription,
        "city": location,
        "forecast_days": forecast_days,
        "forecast": forecast_df.to_dict(orient="records")
    }
    logs.append(entry)
    logging.info(f"Log stocké: {entry}")

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# -------------------- Lancement du backend en arrière-plan --------------------
if "backend_started" not in st.session_state:
    st.session_state.backend_started = True
    threading.Thread(target=run_backend, daemon=True).start()
    time.sleep(1)

# -------------------- Interface Streamlit (Frontend) --------------------
st.title("Application Météo – Commande vocale / manuelle (Open-Meteo)")

# Deux onglets : Prévisions et Analyse & Monitoring
tab1, tab2 = st.tabs(["Prévisions (Python)", "Analyse & Monitoring"])

with tab1:
    st.header("Envoi de la commande au backend")
    mode = st.radio("Choisissez votre mode de commande :", 
                    ("Enregistrement par micro", "Charger un fichier audio", "Manuelle"))
    
    audio_bytes = None
    transcription_input = ""
    city_input = ""
    # La zone "Horizon temporel" est supprimée
    forecast_days_input = None
    if mode != "Enregistrement par micro":
        forecast_days_input = st.selectbox("Nombre de jours de prévision", options=[3, 5, 7], index=2)
    
    if mode == "Enregistrement par micro":
        st.subheader("Enregistrement par micro")
        webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=256,
            client_settings={
                "RTCConfiguration": {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                "mediaStreamConstraints": {"audio": True, "video": False},
            },
            audio_processor_factory=lambda: AudioRecorder()
        )
        if webrtc_ctx.audio_processor:
            if st.button("Arrêter l'enregistrement"):
                audio_bytes = webrtc_ctx.audio_processor.get_audio_bytes()
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    st.info("Enregistrement capturé.")
                else:
                    st.warning("Aucune donnée audio capturée.")
        transcription_input = st.text_input("Transcription (optionnelle)")
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
    
    if st.button("Envoyer la commande"):
        backend_url = "http://localhost:8000/process_command"
        files = {}
        data = {}
        if audio_bytes is not None:
            files["file"] = ("audio.wav", audio_bytes, "audio/wav")
        if transcription_input:
            data["transcription"] = transcription_input
        if city_input:
            data["city"] = city_input
        # En mode manuel, on envoie le nombre de jours choisi
        if mode != "Enregistrement par micro" and forecast_days_input is not None:
            data["forecast_days"] = str(forecast_days_input)
        try:
            response = requests.post(backend_url, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prévision pour {result['location']}")
                final_days = result["forecast_days"]
                hourly_list = result["forecast"]["hourly"]
                df = pd.DataFrame(hourly_list)
                df['date'] = pd.to_datetime(df['date'])
                df['hour'] = df['date'].dt.hour
                df_filtered = df[df['hour'] == 12].sort_values(by='date').head(final_days)
                
                # Graphique interactif avec Plotly
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    subplot_titles=("Température (°C)", "Ciel (%)", "Vent (km/h)"))
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['temperature_2m'],
                                         mode='lines+markers', name='Température', marker=dict(color='red')),
                                         row=1, col=1)
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['cloudcover'],
                                         mode='lines+markers', name='Ciel', marker=dict(color='blue')),
                                         row=2, col=1)
                fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['windspeed_10m'],
                                         mode='lines+markers', name='Vent', marker=dict(color='green')),
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
            else:
                st.error(f"Erreur lors de la récupération de la prévision (code {response.status_code})")
        except Exception as e:
            st.error("Impossible de joindre le backend. Vérifiez qu'il est démarré et accessible.")
    
with tab2:
    st.header("Analyse et Monitoring")
    analysis_url = "http://localhost:8000/analysis"
    try:
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
