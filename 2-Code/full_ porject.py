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

# Pour le caching et retry lors des appels à Open-Meteo
import requests_cache
import pandas as pd
from retry_requests import retry

# Pour intégrer du code HTML/JS dans Streamlit
import streamlit.components.v1 as components

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

# Stockage en mémoire pour le monitoring (simulation d'une base)
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
        # 1. Transcription (simulation si audio fourni)
        if file is not None:
            audio_bytes = await file.read()
            transcription_result = azure_speech_to_text(audio_bytes)
            transcription_final = transcription_result
        else:
            transcription_final = transcription or ""
        
        # 2. Extraction automatique via spaCy (ville et horizon)
        if transcription_final:
            extracted_city, extracted_horizon = spacy_analyze(transcription_final)
        else:
            extracted_city, extracted_horizon = (None, None)
        
        # 3. Priorité aux saisies manuelles
        final_city = city if city else (extracted_city if extracted_city else "Paris")
        final_horizon = horizon if horizon else (extracted_horizon if extracted_horizon else "demain")
        
        # 4. Récupération de la prévision météo via Open-Meteo (hourly)
        hourly_dataframe = get_weather_forecast(final_city)
        logging.info(f"Prévision météo récupérée (hourly) pour {final_city}")
        
        # 5. Stockage dans les logs
        store_forecast_in_db(transcription_final, final_city, final_horizon, hourly_dataframe)
        
        forecast_dict = hourly_dataframe.to_dict(orient="records")
        return WeatherResponse(
            location=final_city,
            forecast={"hourly": forecast_dict},
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
    # Simulation d'appel à Azure Speech-to-Text (à remplacer par l'intégration réelle)
    return "transcrire l'audio: Paris demain"

from typing import Tuple

def spacy_analyze(text: str) -> Tuple[str, str]:
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

# -------------------- Géocodage via Nominatim --------------------
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

# -------------------- Récupération de la prévision météo via Open-Meteo --------------------
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

def store_forecast_in_db(transcription: str, location: str, horizon: str, forecast_df: pd.DataFrame):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transcription": transcription,
        "city": location,
        "horizon": horizon,
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

# Onglets de navigation
tab1, tab2, tab3 = st.tabs(["Prévisions (Python)", "Analyse & Monitoring", "Dashboard HTML/JS"])

with tab1:
    st.header("Envoi de la commande au backend")
    mode = st.radio("Choisissez votre mode de commande :", 
                    ("Enregistrement par micro", "Charger un fichier audio", "Manuelle"))
    
    audio_bytes = None
    transcription_input = ""
    city_input = ""
    horizon_input = ""
    
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
        horizon_input = st.text_input("Horizon temporel (optionnel)")
    
    elif mode == "Charger un fichier audio":
        st.subheader("Charger un fichier audio")
        uploaded_file = st.file_uploader("Sélectionnez votre fichier audio", type=["wav", "mp3"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes, format="audio/wav")
        transcription_input = st.text_input("Transcription (optionnelle)")
        city_input = st.text_input("Ville (optionnel)")
        horizon_input = st.text_input("Horizon temporel (optionnel)")
    else:
        st.subheader("Commande manuelle")
        transcription_input = st.text_input("Transcription de la commande (facultatif)")
        city_input = st.text_input("Ville")
        horizon_input = st.text_input("Horizon temporel (ex. demain, aujourd'hui, etc.)")
    
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
        if horizon_input:
            data["horizon"] = horizon_input
        try:
            response = requests.post(backend_url, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prévision pour {result['location']}")
                hourly_list = result["forecast"]["hourly"]
                df = pd.DataFrame(hourly_list)
                df['date'] = pd.to_datetime(df['date'])
                df['hour'] = df['date'].dt.hour
                # Filtrer pour n'afficher que la prévision de midi (12h00) pour 7 jours
                df_filtered = df[df['hour'] == 12].sort_values(by='date').head(7)
                st.subheader("Prévisions de Midi sur 7 Jours")
                cols = st.columns(len(df_filtered))
                for idx, (_, row) in enumerate(df_filtered.iterrows()):
                    with cols[idx]:
                        st.markdown(f"**{row['date'].strftime('%A %d %b')}**")
                        st.metric(label="🌡️ Température", value=f"{row['temperature_2m']:.1f} °C")
                        st.metric(label="☁️ Ciel", value=f"{row['cloudcover']:.0f} %")
                        st.metric(label="💨 Vent", value=f"{row['windspeed_10m']:.1f} km/h")
                        st.metric(label="😷 Pollution", value=f"{row['pm2_5']:.1f} µg/m³")
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
    
with tab3:
    st.header("Dashboard HTML/JS")
    # Dashboard autonome en HTML affichant uniquement la prévision de midi pour 7 jours
    html_dashboard = """
    <!DOCTYPE html>
    <html lang="fr">
      <head>
        <meta charset="UTF-8" />
        <title>Dashboard Météo – Prévisions de Midi sur 7 Jours</title>
        <style>
          body {
            font-family: Arial, sans-serif;
            background: #f0f4f7;
            margin: 0;
            padding: 20px;
          }
          h1 {
            text-align: center;
            color: #333;
          }
          #dashboard {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
          }
          .card {
            background: #fff;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 180px;
            text-align: center;
          }
          .card h2 {
            margin: 0 0 10px;
            font-size: 1.1em;
            color: #0077cc;
          }
          .card p {
            margin: 6px 0;
            font-size: 0.95em;
          }
        </style>
      </head>
      <body>
        <h1>Prévisions de Midi sur 7 Jours</h1>
        <div id="dashboard">Chargement des prévisions…</div>
        <script type="module">
          import { fetchWeatherApi } from 'https://cdn.jsdelivr.net/npm/openmeteo@1.0.2/dist/openmeteo.mjs';
          // Fonction utilitaire pour générer une plage de nombres
          const range = (start, stop, step) =>
            Array.from({ length: Math.floor((stop - start) / step) }, (_, i) => start + i * step);
          async function main() {
            // Paramètres pour récupérer les prévisions horaires depuis Open-Meteo pour Paris
            const params = {
              latitude: 48.8534,
              longitude: 2.3488,
              hourly: ["temperature_2m", "cloudcover", "windspeed_10m"],
              timezone: "auto"
            };
            const url = "https://api.open-meteo.com/v1/forecast";
            const responses = await fetchWeatherApi(url, params);
            const response = responses[0];
            const utcOffsetSeconds = response.utcOffsetSeconds();
            const hourly = response.hourly();
            const interval = hourly.interval();
            const startTime = Number(hourly.time());
            const endTime = Number(hourly.timeEnd());
            const times = range(startTime, endTime, interval).map(
              t => new Date((t + utcOffsetSeconds) * 1000)
            );
            const hourlyTemp = hourly.variables(0).valuesArray();
            const hourlyCloudCover = hourly.variables(1).valuesArray();
            const hourlyWind = hourly.variables(2).valuesArray();
            // Filtrer pour obtenir uniquement la prévision pour midi (12h00)
            const forecasts = {};
            for (let i = 0; i < times.length; i++) {
              if (times[i].getHours() === 12) {
                const dateKey = times[i].toISOString().split("T")[0];
                forecasts[dateKey] = {
                  time: times[i],
                  temperature: hourlyTemp[i],
                  cloudCover: hourlyCloudCover[i],
                  windSpeed: hourlyWind[i]
                };
              }
            }
            // Transformer l'objet en tableau et ne garder que les 7 premiers jours
            const forecastArray = Object.values(forecasts).slice(0, 7);
            const dashboardDiv = document.getElementById("dashboard");
            dashboardDiv.innerHTML = "";
            forecastArray.forEach(forecast => {
              const card = document.createElement("div");
              card.className = "card";
              const options = { weekday: 'long', day: 'numeric', month: 'short' };
              const dateStr = forecast.time.toLocaleDateString("fr-FR", options);
              card.innerHTML = `
                <h2>${dateStr}</h2>
                <p>🌡️ Température : ${forecast.temperature} °C</p>
                <p>☁️ Ciel : ${forecast.cloudCover} %</p>
                <p>💨 Vent : ${forecast.windSpeed} km/h</p>
              `;
              dashboardDiv.appendChild(card);
            });
          }
          main().catch(error => {
            console.error("Erreur dans le dashboard :", error);
            document.getElementById("dashboard").innerText = "Erreur de chargement des prévisions.";
          });
        </script>
      </body>
    </html>
    """
    components.html(html_dashboard, height=600, scrolling=True)
