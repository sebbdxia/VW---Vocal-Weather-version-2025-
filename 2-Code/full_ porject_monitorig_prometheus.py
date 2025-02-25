import os
import json
import datetime
import logging
import spacy
import requests
import uvicorn
import threading
import time
import re
import tempfile
from functools import wraps
from typing import Tuple, Callable

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Request
from pydantic import BaseModel

import streamlit as st
import requests_cache
import pandas as pd
from retry_requests import retry

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

# Configuration PostgreSQL (Azure)
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Configuration Azure Speech
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")

# Création d'un registre Prometheus personnalisé et ajout des collecteurs système
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
prom_registry = CollectorRegistry(auto_describe=True)
REQUEST_COUNT = Counter("http_requests_total", "Nombre total de requêtes HTTP", ["method", "endpoint", "http_status"], registry=prom_registry)
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Durée des requêtes HTTP (en secondes)", ["method", "endpoint"], registry=prom_registry)
FORECAST_REQUESTS = Counter("forecast_requests_total", "Nombre total de demandes de prévisions traitées", registry=prom_registry)
ERRORS_COUNT = Counter("errors_total", "Nombre total d'erreurs survenues", registry=prom_registry)
FEEDBACK_COUNT = Counter("feedback_total", "Nombre total de retours utilisateurs enregistrés", registry=prom_registry)

from prometheus_client import PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR
prom_registry.register(PROCESS_COLLECTOR)
prom_registry.register(PLATFORM_COLLECTOR)
prom_registry.register(GC_COLLECTOR)

# Session HTTP avec cache et mécanisme de retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# Initialisation de FastAPI et du modèle NLP
app = FastAPI()
logging.basicConfig(level=logging.INFO)
nlp = spacy.load("fr_core_news_sm")
logs = []
user_feedbacks = []

# Middleware Prometheus
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_time = time.time() - start_time
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(elapsed_time)
    return response

def measure_latency(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        logging.info(f"Temps d'exécution de {func.__name__} : {time.time() - start:.2f} secondes")
        return result
    return wrapper

class WeatherResponse(BaseModel):
    location: str
    forecast: dict
    forecast_days: int
    message: str = None
    transcription: str = None
    mode: str = None

@app.post("/process_command", response_model=WeatherResponse)
@measure_latency
async def process_command(
    file: UploadFile = File(None),
    transcription: str = Form(None),
    city: str = Form(None),
    forecast_days: str = Form(None)
):
    try:
        if file is not None:
            audio_bytes = await file.read()
            transcription_text = azure_speech_to_text(audio_bytes)
        else:
            transcription_text = transcription or ""
        
        extracted_city, extracted_days = spacy_analyze(transcription_text)
        final_city = city if city else extracted_city
        
        if (file is not None) or (transcription_text and not forecast_days):
            final_forecast_days = extracted_days
        else:
            try:
                final_forecast_days = int(forecast_days) if forecast_days is not None else 7
            except ValueError:
                final_forecast_days = 7
        
        hourly_dataframe = get_weather_forecast(final_city)
        mode_used = "vocale" if file is not None or (transcription_text and forecast_days is None) else "manuel"
        store_forecast_in_db(transcription_text, final_city, final_forecast_days, hourly_dataframe, mode_used)
        FORECAST_REQUESTS.inc()
        forecast_data = hourly_dataframe.to_dict(orient="records")
        
        return WeatherResponse(
            location=final_city,
            forecast={"hourly": forecast_data},
            forecast_days=final_forecast_days,
            message="Prévision obtenue avec succès.",
            transcription=transcription_text,
            mode=mode_used
        )
    except Exception as e:
        ERRORS_COUNT.inc()
        logging.error("Erreur lors du traitement de la commande", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la commande")

@app.get("/analysis")
def analysis():
    return {"total_requests": len(logs), "logs": logs, "feedbacks": user_feedbacks}

@app.get("/metrics")
def metrics():
    try:
        data = generate_latest(prom_registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logging.error("Erreur lors de l'exposition des métriques", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de l'exposition des métriques")

@app.get("/top_cities")
def top_cities():
    city_counts = {}
    for entry in logs:
        city = entry.get("city", "Inconnu")
        city_counts[city] = city_counts.get(city, 0) + 1
    return city_counts

@app.get("/feedbacks")
def get_feedbacks():
    return user_feedbacks

@app.post("/feedback")
def feedback(rating: int = Form(...), comment: str = Form("")):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rating": rating,
        "comment": comment
    }
    user_feedbacks.append(entry)
    FEEDBACK_COUNT.inc()
    try:
        import psycopg2
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ,
                rating INTEGER,
                comment TEXT
            );
        """)
        cur.execute("""
            INSERT INTO feedback (timestamp, rating, comment)
            VALUES (%s, %s, %s)
        """, (entry["timestamp"], entry["rating"], entry["comment"]))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du stockage du feedback PostgreSQL: {e}")
    return {"status": "Feedback enregistré"}

def azure_speech_to_text(audio_bytes: bytes) -> str:
    import azure.cognitiveservices.speech as speechsdk
    if not SPEECH_KEY or not SPEECH_REGION:
        logging.error("Clés Azure Speech non configurées.")
        return ""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_filename = temp_audio.name
    audio_config = speechsdk.audio.AudioConfig(filename=temp_filename)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once_async().get()
    os.remove(temp_filename)
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        logging.error(f"Transcription annulée: {details.reason}. Détails: {details.error_details}")
        return ""
    else:
        logging.error(f"Erreur de transcription: {result.reason}")
        return ""

def azure_speech_from_microphone() -> str:
    import azure.cognitiveservices.speech as speechsdk
    if not SPEECH_KEY or not SPEECH_REGION:
        logging.error("Les clés d'API Azure Speech ne sont pas configurées.")
        return ""
    
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    # Reconnaissance continue pour une meilleure qualité de transcription
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    st.info("Veuillez parler... (enregistrement continu pendant 10 secondes)")
    
    result_texts = []
    def recognized_cb(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            result_texts.append(evt.result.text)
    def stop_cb(evt):
        st.info("Fin de l'enregistrement.")
    
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)
    
    speech_recognizer.start_continuous_recognition()
    time.sleep(10)
    speech_recognizer.stop_continuous_recognition()
    
    final_text = " ".join(result_texts)
    if not final_text:
        logging.error("Aucune parole détectée lors de l'enregistrement continu.")
    return final_text

def spacy_analyze(text: str) -> Tuple[str, int]:
    doc = nlp(text)
    location = None
    forecast_days = None
    # Recherche des motifs "sur X jours" ou "pour X jours"
    regex_match = re.search(r"(?:sur|pour)\s+(\d+)\s+jours", text, re.IGNORECASE)
    if regex_match:
        try:
            num = int(regex_match.group(1))
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
                except Exception:
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

def store_forecast_in_db(transcription: str, location: str, forecast_days: int, forecast_df: pd.DataFrame, mode: str):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transcription": transcription,
        "city": location,
        "forecast_days": forecast_days,
        "forecast": forecast_df.to_dict(orient="records"),
        "mode": mode
    }
    logs.append(entry)
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
                forecast JSONB,
                mode TEXT
            );
        """)
        cur.execute("""
            INSERT INTO forecasts (timestamp, transcription, city, forecast_days, forecast, mode)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (entry["timestamp"], entry["transcription"], entry["city"], entry["forecast_days"], json.dumps(entry["forecast"]), entry["mode"]))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du stockage en base de données : {e}")

def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if "backend_started" not in st.session_state:
    st.session_state.backend_started = True
    threading.Thread(target=run_backend, daemon=True).start()
    time.sleep(1)

# Interface Streamlit
st.title("Application Météo – Commande vocale et manuelle (Open-Meteo)")

if "forecast_response" not in st.session_state:
    st.session_state["forecast_response"] = None

# Mise à jour des options : seulement "Enregistrement par micro" et "Manuelle"
tab1, tab2, tab3 = st.tabs(["Prévisions", "Analyse & Monitoring", "Feedback"])

with tab1:
    st.header("Bienvenue sur l'application météo")
    mode = st.radio("Sélectionnez le mode de commande :", ("Enregistrement par micro", "Manuelle"))
    transcription_input = ""
    city_input = ""
    forecast_days_input = None

    if mode == "Enregistrement par micro":
        st.subheader("Commande vocale via microphone")
        if st.button("Enregistrer la commande vocale"):
            transcription_result = azure_speech_from_microphone()
            if transcription_result:
                st.session_state.micro_transcription = transcription_result
            else:
                st.error("Enregistrement échoué ou aucune parole détectée. Veuillez réessayer.")
        if "micro_transcription" in st.session_state and st.session_state.micro_transcription:
            st.write("Transcription obtenue :", st.session_state.micro_transcription)
            city_input = st.text_input("Indiquez la ville (facultatif)")
            if st.button("Envoyer la commande vocale"):
                backend_url = "http://localhost:8000/process_command"
                data = {"transcription": st.session_state.micro_transcription, "mode": "vocale"}
                if city_input:
                    data["city"] = city_input
                response = requests.post(backend_url, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.forecast_response = result
                    st.success(f"Prévision pour {result['location']} (commande vocale)")
                else:
                    st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande vocale")
    else:
        st.subheader("Commande manuelle")
        city_input = st.text_input("Ville")
        forecast_days_input = st.selectbox("Nombre de jours de prévision", options=[3, 5, 7], index=2)
        transcription_input = st.text_input("Saisissez votre commande (ex : prévisions pour 3 jours)")
        if st.button("Envoyer la commande"):
            backend_url = "http://localhost:8000/process_command"
            data = {}
            if transcription_input:
                data["transcription"] = transcription_input
            if city_input:
                data["city"] = city_input
            if forecast_days_input is not None:
                data["forecast_days"] = str(forecast_days_input)
            data["mode"] = "manuel"
            try:
                response = requests.post(backend_url, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prévision pour {result['location']}")
                    st.session_state.forecast_response = result
                else:
                    st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande")
            except Exception as e:
                st.error("Impossible de joindre le backend. Vérifiez qu'il est démarré et accessible.")
    
    if st.session_state.forecast_response and st.button("Afficher les résultats"):
        result = st.session_state.forecast_response
        st.subheader("Prévisions")
        final_days = result["forecast_days"]
        df = pd.DataFrame(result["forecast"]["hourly"])
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        # Filtrer pour n'afficher que la prévision à 12h sur le nombre de jours demandé
        df_filtered = df[df['hour'] == 12].sort_values(by='date').head(final_days)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Température (°C)", "Nébulosité (%)", "Vent (km/h)"))
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['temperature_2m'],
                                 mode='lines+markers', marker=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['cloudcover'],
                                 mode='lines+markers', marker=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['windspeed_10m'],
                                 mode='lines+markers', marker=dict(color='green')), row=3, col=1)
        fig.update_layout(height=600, title=f"Prévisions de Midi sur {final_days} jours", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Détails des prévisions")
        st.dataframe(df_filtered[['date', 'temperature_2m', 'cloudcover', 'windspeed_10m', 'pm2_5']].rename(
            columns={
                "date": "Date",
                "temperature_2m": "Température (°C)",
                "cloudcover": "Nébulosité (%)",
                "windspeed_10m": "Vent (km/h)",
                "pm2_5": "Pollution (µg/m³)"
            }
        ))
        st.subheader("Localisation")
        try:
            lat, lon = get_coordinates(result["location"])
            map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
            st.map(map_data)
        except Exception as e:
            st.error(f"Impossible d'afficher la carte: {e}")
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
            st.write("Nombre de retours utilisateurs :", len(data_analysis.get("feedbacks", [])))
        else:
            st.error("Erreur lors de la récupération des données d'analyse.")
    except Exception as e:
        st.error("Impossible de joindre le backend pour l'analyse.")
    
    try:
        metrics_url = "http://localhost:8000/metrics"
        metrics_response = requests.get(metrics_url)
        if metrics_response.status_code == 200:
            lines = metrics_response.text.splitlines()
            metrics_list = []
            for line in lines:
                if line.startswith("#") or line.strip() == "":
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    metric_name = parts[0]
                    metric_value = parts[1]
                    metrics_list.append({"Metric": metric_name, "Value": metric_value})
            # Liste des 10 métriques les plus pertinentes
            relevant_metric_names = [
                "http_requests_total",
                "http_request_duration_seconds",
                "forecast_requests_total",
                "errors_total",
                "feedback_total",
                "process_cpu_seconds_total",
                "process_start_time_seconds",
                "process_resident_memory_bytes",
                "python_gc_objects_collected_total",
                "python_gc_objects_uncollectable_total"
            ]
            filtered_metrics = [m for m in metrics_list if m["Metric"] in relevant_metric_names]
            df_metrics = pd.DataFrame(filtered_metrics)
            st.subheader("Métriques Prometheus")
            st.dataframe(df_metrics)
            
            definitions = {
                "http_requests_total": "Nombre total de requêtes HTTP, ventilées par méthode, endpoint et code HTTP.",
                "http_request_duration_seconds": "Durée des requêtes HTTP (en secondes) par méthode et endpoint.",
                "forecast_requests_total": "Nombre total de demandes de prévisions traitées.",
                "errors_total": "Nombre total d'erreurs survenues lors du traitement des requêtes.",
                "feedback_total": "Nombre total de retours utilisateurs enregistrés.",
                "process_cpu_seconds_total": "Temps CPU utilisé par le processus.",
                "process_start_time_seconds": "Heure de démarrage du processus.",
                "process_resident_memory_bytes": "Mémoire résidente utilisée par le processus (en octets).",
                "python_gc_objects_collected_total": "Nombre total d'objets collectés par le ramasse-miettes Python.",
                "python_gc_objects_uncollectable_total": "Nombre total d'objets non collectables par le ramasse-miettes Python."
            }
            st.markdown("### Définitions des métriques")
            for metric, definition in definitions.items():
                st.markdown(f"- **{metric}** : {definition}")
        else:
            st.error("Erreur lors de la récupération des métriques Prometheus.")
    except Exception as e:
        st.error("Impossible de joindre l'endpoint /metrics.")
    
    try:
        top_cities_url = "http://localhost:8000/top_cities"
        response = requests.get(top_cities_url)
        if response.status_code == 200:
            st.subheader("Répartition des demandes par ville")
            top_cities_data = response.json()
            df_top = pd.DataFrame(list(top_cities_data.items()), columns=["Ville", "Nombre de demandes"])
            st.dataframe(df_top)
            fig = go.Figure(data=[go.Bar(x=df_top["Ville"], y=df_top["Nombre de demandes"], marker_color='indianred')])
            fig.update_layout(title="Nombre de demandes par ville", xaxis_title="Ville", yaxis_title="Nombre de demandes", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Erreur lors de la récupération des données par ville.")
    except Exception as e:
        st.error("Impossible de joindre l'endpoint /top_cities.")

with tab3:
    st.header("Feedback")
    with st.form("feedback_form"):
        rating = st.slider("Votre note (1 à 5)", min_value=1, max_value=5, value=3)
        comment = st.text_area("Votre commentaire (facultatif)")
        submitted = st.form_submit_button("Envoyer le feedback")
        if submitted:
            feedback_url = "http://localhost:8000/feedback"
            data = {"rating": rating, "comment": comment}
            response = requests.post(feedback_url, data=data)
            if response.status_code == 200:
                st.success("Merci pour votre retour.")
            else:
                st.error("Erreur lors de l'envoi du feedback.")
