#!/usr/bin/env python
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
from typing import Tuple, Callable
from functools import wraps

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import streamlit as st
import requests_cache
import pandas as pd
from retry_requests import retry

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dotenv import load_dotenv
load_dotenv('.env')

# ─────────────── Intégration d'OpenTelemetry pour la traçabilité distribuée ───────────────
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

resource = Resource(attributes={"service.name": "weather-app"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# ─────────────── Configuration de la base de données et des métriques Prometheus ───────────────
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
prom_registry = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "endpoint", "http_status"],
    registry=prom_registry
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP (en secondes)",
    ["method", "endpoint"],
    registry=prom_registry
)

FORECAST_REQUESTS = Counter(
    "forecast_requests_total",
    "Nombre total de demandes de prévisions traitées",
    registry=prom_registry
)
ERRORS_COUNT = Counter(
    "errors_total",
    "Nombre total d'erreurs survenues",
    registry=prom_registry
)
FEEDBACK_COUNT = Counter(
    "feedback_total",
    "Nombre total de retours utilisateurs enregistrés",
    registry=prom_registry
)
# Nouveau Gauge pour le taux d'erreur par endpoint
HTTP_ERROR_RATE = Gauge(
    "http_error_rate",
    "Taux d'erreur HTTP par endpoint",
    ["endpoint"],
    registry=prom_registry
)

# ─────────────── Centralisation des logs ───────────────
logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler("aggregated_logs.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

# Sessions en cache et reprise automatique des requêtes HTTP
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# Clés d'accès au service de transcription vocale Azure
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")

# Variables globales pour logs, feedback et statistiques par endpoint
app = FastAPI()
nlp = spacy.load("fr_core_news_sm")
logs = []            
user_feedbacks = []

# Structure pour stocker le nombre total de requêtes et d'erreurs par endpoint
endpoint_stats = {}  # Exemple: {"/process_command": {"total": 10, "errors": 2}, ...}
alerts = []  # Liste des alertes générées

# ─────────────── Middleware pour instrumentation et collecte fine ───────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_time = time.time() - start_time
    path = request.url.path
    method = request.method
    status = response.status_code
    REQUEST_COUNT.labels(method=method, endpoint=path, http_status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed_time)
    
    # Mise à jour des statistiques par endpoint pour le calcul du taux d'erreur
    if path not in endpoint_stats:
         endpoint_stats[path] = {"total": 0, "errors": 0}
    endpoint_stats[path]["total"] += 1
    if status >= 400:
         endpoint_stats[path]["errors"] += 1
         ERRORS_COUNT.inc()
    # Mise à jour du Gauge pour chaque endpoint
    for ep, stats in endpoint_stats.items():
         error_rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0
         HTTP_ERROR_RATE.labels(endpoint=ep).set(error_rate)
    
    return response

# ─────────────── Décorateur pour mesurer la latence et créer des spans OpenTelemetry ───────────────
def measure_latency(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        with tracer.start_as_current_span(func.__name__):
            result = await func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"Temps d'exécution de {func.__name__}: {elapsed:.3f} secondes")
        return result
    return wrapper

# ─────────────── Modèle de réponse et endpoints de l'API ───────────────
class WeatherResponse(BaseModel):
    location: str
    forecast: dict
    forecast_days: int
    forecast_offset: int = 0  # Nouveau champ pour le décalage (0 = aujourd'hui, 1 = demain)
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
            transcription_result = azure_speech_to_text(audio_bytes)
            transcription_final = transcription_result
        else:
            transcription_final = transcription or ""
        
        # Analyse de la commande avec prise en compte des mots-clés "aujourd'hui" et "demain"
        extracted_city, extracted_days, extracted_offset = spacy_analyze(transcription_final)
        logging.info(f"Extraction: Ville='{extracted_city}', Jours='{extracted_days}', Offset='{extracted_offset}'")
        final_city = city if city else extracted_city
        
        if (file is not None) or (transcription_final and not forecast_days):
            final_forecast_days = extracted_days
            final_forecast_offset = extracted_offset
        else:
            final_forecast_days = int(forecast_days) if forecast_days is not None else 7
            final_forecast_offset = 0
        
        hourly_dataframe = get_weather_forecast(final_city)
        logging.info(f"Prévision météo récupérée pour {final_city}")
        store_forecast_in_db(transcription_final, final_city, final_forecast_days, hourly_dataframe, "vocale" if file is not None else "manuel")
        
        FORECAST_REQUESTS.inc()
        forecast_dict = hourly_dataframe.to_dict(orient="records")
        return WeatherResponse(
            location=final_city,
            forecast={"hourly": forecast_dict},
            forecast_days=final_forecast_days,
            forecast_offset=final_forecast_offset,
            message="Prévision obtenue avec succès.",
            transcription=transcription_final,
            mode="vocale" if file is not None else "manuel"
        )
    except Exception as e:
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

# ─────────────── Fonctions de traitement et intégration externe ───────────────
def azure_speech_to_text(audio_bytes: bytes = None) -> str:
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

def spacy_analyze(text: str) -> Tuple[str, int, int]:
    doc = nlp(text)
    location = None
    forecast_days = None
    offset = 0
    lower_text = text.lower()
    # Prise en compte des mots-clés pour "aujourd'hui" et "demain"
    if "aujourd'hui" in lower_text and "demain" in lower_text:
         forecast_days = 2
         offset = 0
    elif "aujourd'hui" in lower_text:
         forecast_days = 1
         offset = 0
    elif "demain" in lower_text:
         forecast_days = 1
         offset = 1  # Décalage pour afficher la prévision de demain uniquement.
    
    # Recherche de pattern numérique si aucun mot-clé n'est détecté
    if forecast_days is None:
         match = re.search(r"sur\s+(\d+)\s+jours", text, re.IGNORECASE)
         if match:
             try:
                 num = int(match.group(1))
                 if num in [3, 5, 7]:
                     forecast_days = num
                     offset = 0
             except Exception as e:
                 logging.error(f"Erreur extraction jours: {e}")
    if forecast_days is None:
         for token in doc:
             if token.like_num:
                 try:
                     num = int(token.text)
                     if num in [3, 5, 7]:
                         forecast_days = num
                         offset = 0
                         break
                 except:
                     continue
    for ent in doc.ents:
         if ent.label_ in ["LOC", "GPE"] and not location:
              location = ent.text
    if not location:
         location = "Paris"
    if forecast_days is None:
         forecast_days = 7
         offset = 0
    return location, forecast_days, offset

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
    logging.info(f"Log stocké: {entry}")
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
        logging.error(f"Erreur lors du stockage PostgreSQL: {e}")

# ─────────────── Fonction de vérification et génération d'alertes ───────────────
def check_alerts():
    global alerts
    alerts.clear()
    threshold = 0.1  # Seuil fixé à 10%
    for ep, stats in endpoint_stats.items():
         error_rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0
         if error_rate > threshold:
              alerts.append(f"Attention : Taux d'erreur élevé sur {ep} ({error_rate:.2%})")

# ─────────────── Démarrage du Backend et intégration avec l'interface Streamlit ───────────────
def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if "backend_started" not in st.session_state:
    st.session_state.backend_started = True
    threading.Thread(target=run_backend, daemon=True).start()
    time.sleep(1)

st.title("Application Météo – Commande vocale / manuelle (Open-Meteo)")

if "forecast_response" not in st.session_state:
    st.session_state["forecast_response"] = None

# Construction de l'interface en trois onglets.
tab1, tab2, tab3 = st.tabs(["Prévisions", "Analyse & Monitoring", "Feedback"])

with tab1:
    st.header("Bienvenu !")
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
            backend_url = "http://localhost:8000/process_command"
            data = {"transcription": transcription_input, "mode": "vocale"}
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
        else:
            st.subheader("Commande manuelle")
        
    
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
        data["mode"] = "manuel"
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
    
    if st.session_state.forecast_response and st.button("Afficher les résultats de la commande vocale"):
        result = st.session_state.forecast_response
        st.subheader("Prévisions actuelles")
        final_days = result["forecast_days"]
        offset = result.get("forecast_offset", 0)
        df = pd.DataFrame(result["forecast"]["hourly"])
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        # Filtre sur l'heure de midi puis application du décalage
        df_filtered = df[df['hour'] == 12].sort_values(by='date')
        if offset > 0:
            df_filtered = df_filtered.iloc[offset:offset+final_days]
        else:
            df_filtered = df_filtered.head(final_days)
        
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
        st.subheader("Localisation de la ville")
        try:
            lat, lon = get_coordinates(result["location"])
            map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
            st.map(map_data)
        except Exception as e:
            st.error(f"Impossible d'afficher la carte: {e}")
        if result.get("transcription"):
            st.write("Transcription utilisée :", result["transcription"])

with tab2:
    st.header("Analyse & Monitoring")
    try:
        analysis_url = "http://localhost:8000/analysis"
        response = requests.get(analysis_url)
        if response.status_code == 200:
            data_analysis = response.json()
            st.write("Nombre total de requêtes :", data_analysis.get("total_requests", 0))
            st.write("Retours utilisateurs :", len(data_analysis.get("feedbacks", [])))
        else:
            st.error("Erreur lors de la récupération des données d'analyse.")
    except Exception as e:
        st.error("Impossible de joindre le backend pour l'analyse.")
    
    try:
        metrics_url = "http://localhost:8000/metrics"
        metrics_response = requests.get(metrics_url)
        if metrics_response.status_code == 200:
            st.subheader("Métriques Prometheus")
            lines = metrics_response.text.splitlines()
            definitions = {
                "http_requests_total": "Nombre total de requêtes HTTP, ventilées par méthode, endpoint et code HTTP.",
                "http_request_duration_seconds": "Durée des requêtes HTTP par méthode et endpoint.",
                "forecast_requests_total": "Nombre total de demandes de prévisions traitées.",
                "errors_total": "Nombre total d'erreurs survenues.",
                "feedback_total": "Nombre total de retours utilisateurs enregistrés.",
                "http_error_rate": "Taux d'erreur HTTP par endpoint."
            }
            metrics_data = []
            for key, definition in definitions.items():
                for line in lines:
                    if line.startswith(key):
                        parts = line.split()
                        if len(parts) >= 2:
                            metric_value = parts[1]
                            metrics_data.append({"Metric": key, "Value": metric_value})
                        break
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                df_metrics_styled = df_metrics.style.set_properties(subset=["Value"], **{'min-width': '300px'})
                st.dataframe(df_metrics_styled)
                st.markdown("**Définitions des métriques :**")
                for key, definition in definitions.items():
                    st.markdown(f"- **{key}** : {definition}")
            else:
                st.write("Aucune métrique trouvée.")
        else:
            st.error("Erreur lors de la récupération des métriques Prometheus.")
    except Exception as e:
        st.error("Impossible de joindre l'endpoint /metrics.")
    
    # Affichage des alertes générées
    st.subheader("Alertes")
    check_alerts()
    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("Aucune alerte détectée.")
    
    # Affichage du taux d'erreur par endpoint
    st.subheader("Taux d'erreur par endpoint")
    data_error_rate = []
    for ep, stats in endpoint_stats.items():
         error_rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0
         data_error_rate.append({
             "Endpoint": ep,
             "Total Requests": stats["total"],
             "Errors": stats["errors"],
             "Error Rate": f"{error_rate:.2%}"
         })
    if data_error_rate:
         df_error = pd.DataFrame(data_error_rate)
         st.dataframe(df_error)
    else:
         st.write("Aucune donnée d'erreur disponible.")
    
    # Affichage des logs centralisés (les 10 dernières lignes)
    st.subheader("Logs centralisés")
    try:
        with open("aggregated_logs.log", "r") as f:
             logs_content = f.read().splitlines()[-10:]
        for line in logs_content:
             st.text(line)
    except Exception as e:
        st.error("Impossible de lire les logs centralisés.")

with tab3:
    st.header("Feedback")
    # Formulaire de saisie de feedback
    with st.form("feedback_form"):
        rating = st.slider("Votre note (1-5)", min_value=1, max_value=5, value=3)
        comment = st.text_area("Votre commentaire (facultatif)")
        submitted = st.form_submit_button("Envoyer votre feedback")
        if submitted:
            feedback_url = "http://localhost:8000/feedback"
            data = {"rating": rating, "comment": comment}
            response = requests.post(feedback_url, data=data)
            if response.status_code == 200:
                st.success("Merci pour votre feedback !")
            else:
                st.error("Erreur lors de l'envoi du feedback.")

    # Récupération et affichage des feedbacks existants
    try:
        feedbacks_response = requests.get("http://localhost:8000/feedbacks")
        if feedbacks_response.status_code == 200:
            feedbacks = feedbacks_response.json()
            if feedbacks:
                # Calcul de la note moyenne
                average_rating = sum(fb["rating"] for fb in feedbacks) / len(feedbacks)
                st.markdown(f"**Note moyenne : {average_rating:.2f} / 5**")
                
                # Visualisation de la répartition des notes
                df_feedbacks = pd.DataFrame(feedbacks)
                rating_counts = df_feedbacks["rating"].value_counts().sort_index()
                fig = go.Figure(data=go.Bar(
                    x=rating_counts.index.astype(str),
                    y=rating_counts.values,
                    marker_color="indianred"
                ))
                fig.update_layout(
                    title="Répartition des notes",
                    xaxis_title="Note",
                    yaxis_title="Nombre de feedbacks",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Affichage détaillé des commentaires
                st.subheader("Détails des feedbacks")
                st.dataframe(df_feedbacks[["timestamp", "rating", "comment"]])
            else:
                st.info("Aucun feedback enregistré pour le moment.")
        else:
            st.error("Erreur lors de la récupération des feedbacks.")
    except Exception as e:
        st.error(f"Erreur lors de la connexion au backend pour les feedbacks: {e}")
