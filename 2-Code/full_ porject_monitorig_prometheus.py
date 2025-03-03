import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")

# Remarque : Pour éviter l'erreur "cannot import name 'imaging' from 'PIL'",
# assurez-vous d'installer une version compatible de Pillow (exemple : Pillow==9.5.0)
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
from typing import Tuple, Callable, Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Request
from pydantic import BaseModel

import streamlit as st
import streamlit.components.v1 as components
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

# Initialisation de FastAPI et du modèle NLP (la version full_projet est privilégiée)
app = FastAPI()
logging.basicConfig(level=logging.INFO)
nlp = spacy.load("fr_core_news_sm")

# Liste pour stocker les retours utilisateurs
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

# Modèle de réponse pour les prévisions
class WeatherResponse(BaseModel):
    location: str
    forecast: dict
    forecast_days: int
    message: Optional[str] = None
    transcription: Optional[str] = None
    mode: Optional[str] = None

# Analyse du texte à l'aide de Spacy (version full_projet privilégiée)
def spacy_analyze(text: str) -> Tuple[str, int]:
    doc = nlp(text)
    city = None
    days = 7
    for ent in doc.ents:
        if ent.label_ == "LOC":
            city = ent.text
        elif ent.label_ == "DATE":
            match = re.search(r'\d+', ent.text)
            if match:
                days = int(match.group())
    return city, days

# Fonctions complémentaires issues de la version corrected-weather-app pour améliorer la récupération des prévisions
def get_coordinates(city_name: str) -> Tuple[float, float]:
    try:
        geocode_url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city_name, "format": "json"}
        headers = {"User-Agent": "WeatherApp/1.0"}
        r = requests.get(geocode_url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            logging.warning(f"Ville introuvable : {city_name}. Utilisation de Paris par défaut.")
            return 48.8566, 2.3522
        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logging.error(f"Erreur de géocodage pour {city_name}: {str(e)}", exc_info=True)
        return 48.8566, 2.3522

def get_weather_forecast(city_name: str) -> pd.DataFrame:
    try:
        lat, lon = get_coordinates(city_name)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,cloudcover,windspeed_10m",
            "timezone": "auto"
        }
        response = retry_session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        times = pd.to_datetime(data['hourly']['time'])
        df = pd.DataFrame({
            "date": times,
            "temperature_2m": data['hourly']['temperature_2m'],
            "cloudcover": data['hourly']['cloudcover'],
            "windspeed_10m": data['hourly']['windspeed_10m'],
            "pm2_5": [12.3] * len(times)  # Valeur fictive pour la pollution
        })
        return df
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des prévisions météo: {str(e)}", exc_info=True)
        current_time = datetime.datetime.now()
        dates = [current_time + datetime.timedelta(hours=i) for i in range(7*24)]
        df = pd.DataFrame({
            "date": dates,
            "temperature_2m": [20.0] * len(dates),
            "cloudcover": [50.0] * len(dates),
            "windspeed_10m": [10.0] * len(dates),
            "pm2_5": [12.3] * len(dates)
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
        """, (
            datetime.datetime.now(),
            entry["transcription"],
            entry["city"],
            entry["forecast_days"],
            json.dumps(entry["forecast"], default=str),
            entry["mode"]
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du stockage en base de données : {str(e)}", exc_info=True)

# Fonction de transcription depuis un fichier audio (préférant la version full_projet)
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

# Fonction complémentaire pour la commande vocale depuis le micro (issue de corrected-weather-app)
def azure_speech_from_microphone() -> str:
    try:
        import azure.cognitiveservices.speech as speechsdk
        if not SPEECH_KEY or not SPEECH_REGION:
            logging.error("Les clés d'API Azure Speech ne sont pas configurées.")
            return ""
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
        speech_config.speech_recognition_language = "fr-FR"
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
    except Exception as e:
        logging.error(f"Erreur lors de l'enregistrement audio: {str(e)}", exc_info=True)
        return ""

# Endpoint principal de traitement de la commande
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
        mode_used = "vocale" if file is not None or (transcription_text and not forecast_days) else "manuel"
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

# Endpoint pour récupérer les logs et retours
@app.get("/analysis")
def analysis():
    try:
        import psycopg2
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, transcription, city, forecast_days, forecast, mode 
            FROM forecasts
            ORDER BY timestamp DESC
        """)
        rows = cur.fetchall()
        logs_db = []
        for row in rows:
            logs_db.append({
                "timestamp": row[0],
                "transcription": row[1],
                "city": row[2],
                "forecast_days": row[3],
                "forecast": row[4],
                "mode": row[5]
            })
        cur.close()
        conn.close()
        return {"total_requests": len(logs_db), "logs": logs_db, "feedbacks": user_feedbacks}
    except Exception as e:
        logging.error("Erreur lors de la récupération des logs depuis Azure", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des logs")

# Endpoint d'exposition des métriques Prometheus
@app.get("/metrics")
def metrics():
    try:
        data = generate_latest(prom_registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logging.error("Erreur lors de l'exposition des métriques", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de l'exposition des métriques")

# Endpoint pour obtenir le nombre de demandes par ville
@app.get("/top_cities")
def top_cities():
    try:
        import psycopg2
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        cur = conn.cursor()
        cur.execute("SELECT city, COUNT(*) FROM forecasts GROUP BY city")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        city_counts = {row[0]: row[1] for row in rows}
        return city_counts
    except Exception as e:
        logging.error("Erreur lors de la récupération des données de villes depuis Azure", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des données")

# Endpoints pour les retours utilisateurs
@app.get("/feedbacks")
def get_feedbacks():
    try:
        import psycopg2
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        cur = conn.cursor()
        cur.execute("SELECT timestamp, rating, comment FROM feedback ORDER BY timestamp DESC")
        rows = cur.fetchall()
        feedbacks = []
        for row in rows:
            feedbacks.append({
                "timestamp": row[0].isoformat() if isinstance(row[0], datetime.datetime) else row[0],
                "rating": row[1],
                "comment": row[2]
            })
        cur.close()
        conn.close()
        return feedbacks
    except Exception as e:
        logging.error("Erreur lors de la récupération des feedbacks: " + str(e), exc_info=True)
        return user_feedbacks

@app.post("/feedback")
def feedback(rating: int = Form(...), comment: str = Form("")):
    if not 1 <= rating <= 5:
        raise HTTPException(status_code=400, detail="La note doit être comprise entre 1 et 5")
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
        """, (datetime.datetime.now(), entry["rating"], entry["comment"]))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "Feedback enregistré avec succès"}
    except Exception as e:
        logging.error(f"Erreur lors du stockage du feedback PostgreSQL: {e}", exc_info=True)
        return {"status": "Feedback enregistré en mémoire uniquement", "error": str(e)}

# Fonctions d'encodage pour générer l'URL PlantUML
import zlib
import base64

def encode6bit(b: int) -> str:
    if b < 10:
        return chr(48 + b)
    b -= 10
    if b < 26:
        return chr(65 + b)
    b -= 26
    if b < 26:
        return chr(97 + b)
    b -= 26
    if b == 0:
        return '-'
    if b == 1:
        return '_'
    return '?'

def encode64(data: bytes) -> str:
    res = ""
    i = 0
    length = len(data)
    while i < length:
        if i + 3 <= length:
            b1, b2, b3 = data[i], data[i+1], data[i+2]
            i += 3
        elif i + 2 == length:
            b1, b2 = data[i], data[i+1]
            b3 = 0
            i += 2
        else:
            b1 = data[i]
            b2 = 0
            b3 = 0
            i += 1
        c1 = b1 >> 2
        c2 = ((b1 & 3) << 4) | (b2 >> 4)
        c3 = ((b2 & 15) << 2) | (b3 >> 6)
        c4 = b3 & 63
        res += encode6bit(c1) + encode6bit(c2) + encode6bit(c3) + encode6bit(c4)
    return res

def encode_plantuml(plantuml_text: str) -> str:
    compressed = zlib.compress(plantuml_text.encode('utf-8'))
    compressed = compressed[2:-4]
    return encode64(compressed)

def generate_functional_schema() -> str:
    uml_text = """
@startuml
skinparam monochrome true

package "Frontend" {
  [Streamlit App]
}
package "Backend" {
  [FastAPI App]
}
package "Services" {
  [Azure Speech]
  [Open-Meteo API]
  [PostgreSQL]
  [Prometheus]
}

[Streamlit App] --> [FastAPI App] : HTTP Requests
[FastAPI App] --> [PostgreSQL] : Store/Retrieve data
[FastAPI App] --> [Open-Meteo API] : Get weather data
[FastAPI App] --> [Azure Speech] : Speech-to-Text
[FastAPI App] --> [Prometheus] : Metrics collection
@enduml
    """
    encoded = encode_plantuml(uml_text)
    url = f"http://www.plantuml.com/plantuml/png/{encoded}"
    return url

@app.get("/diagram")
def get_diagram():
    try:
        diagram_url = generate_functional_schema()
        return {"diagram_url": diagram_url}
    except Exception as e:
        logging.error("Erreur lors de la génération du schéma fonctionnel", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de la génération du schéma fonctionnel")

# Fonction pour lancer le backend
def run_backend():
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"Erreur lors du démarrage du backend: {str(e)}", exc_info=True)

# Interface Streamlit
st.title("Application Météo – Commande vocale et manuelle (Open-Meteo)")

# Onglets pour naviguer entre les différentes fonctionnalités
tab1, tab2, tab3, tab4 = st.tabs(["Prévisions", "Analyse & Monitoring", "Feedback", "Architecture"])

with tab1:
    st.header("Bienvenue sur l'application météo")
    mode = st.radio("Sélectionnez le mode de commande :", ("Enregistrement par micro", "Manuelle"))
    transcription_input = ""
    city_input = ""
    forecast_days_input = None
    if mode == "Enregistrement par micro":
        st.subheader("Commande vocale via microphone")
        if st.button("Enregistrer la commande vocale"):
            try:
                transcription_result = azure_speech_from_microphone()
                if transcription_result:
                    st.session_state.micro_transcription = transcription_result
                else:
                    st.error("Enregistrement échoué ou aucune parole détectée. Veuillez réessayer.")
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement: {str(e)}")
        if "micro_transcription" in st.session_state and st.session_state.micro_transcription:
            st.write("Transcription obtenue :", st.session_state.micro_transcription)
            city_input = st.text_input("Indiquez la ville (facultatif)")
            if st.button("Envoyer la commande vocale"):
                try:
                    backend_url = "http://localhost:8000/process_command"
                    data = {"transcription": st.session_state.micro_transcription}
                    if city_input:
                        data["city"] = city_input
                    response = requests.post(backend_url, data=data, timeout=30)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.forecast_response = result
                        st.success(f"Prévision pour {result['location']} (commande vocale)")
                    else:
                        st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande vocale")
                except Exception as e:
                    st.error(f"Erreur de communication avec le backend: {str(e)}")
    else:
        st.subheader("Commande manuelle")
        city_input = st.text_input("Ville")
        forecast_days_input = st.selectbox("Nombre de jours de prévision", options=[3, 5, 7], index=2)
        transcription_input = st.text_input("Saisissez votre commande (ex : prévisions pour 3 jours)")
        if st.button("Envoyer la commande"):
            try:
                backend_url = "http://localhost:8000/process_command"
                data = {}
                if transcription_input:
                    data["transcription"] = transcription_input
                if city_input:
                    data["city"] = city_input
                if forecast_days_input is not None:
                    data["forecast_days"] = str(forecast_days_input)
                response = requests.post(backend_url, data=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prévision pour {result['location']}")
                    st.session_state.forecast_response = result
                else:
                    st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande")
            except Exception as e:
                st.error(f"Impossible de joindre le backend. Vérifiez qu'il est démarré et accessible: {str(e)}")
    if "forecast_response" in st.session_state and st.session_state.forecast_response:
        if st.button("Afficher les résultats", key="display_results"):
            try:
                result = st.session_state.forecast_response
                st.subheader("Prévisions")
                final_days = result["forecast_days"]
                df = pd.DataFrame(result["forecast"]["hourly"])
                df['date'] = pd.to_datetime(df['date'])
                df['hour'] = df['date'].dt.hour
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
            except Exception as e:
                st.error(f"Erreur lors de l'affichage des résultats: {str(e)}")

with tab2:
    st.header("Logs et métriques")
    try:
        analysis_url = "http://localhost:8000/analysis"
        metrics_url = "http://localhost:8000/metrics"
        analysis_response = requests.get(analysis_url, timeout=30).json()
        st.write("Total des requêtes:", analysis_response.get("total_requests"))
        st.write("Logs:", analysis_response.get("logs"))
        st.write("Feedbacks:", analysis_response.get("feedbacks"))
        st.write("Métriques Prometheus:", requests.get(metrics_url, timeout=30).text)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {str(e)}")

with tab3:
    st.header("Feedback")
    rating = st.slider("Note (1 à 5)", 1, 5, 3)
    comment = st.text_area("Votre commentaire")
    if st.button("Envoyer le feedback"):
        try:
            feedback_url = "http://localhost:8000/feedback"
            response = requests.post(feedback_url, data={"rating": rating, "comment": comment}, timeout=30)
            if response.status_code == 200:
                st.success("Feedback enregistré")
            else:
                st.error("Erreur lors de l'envoi du feedback")
        except Exception as e:
            st.error(f"Erreur lors de l'envoi du feedback: {str(e)}")

with tab4:
    st.header("Schéma fonctionnel")
    try:
        diagram_response = requests.get("http://localhost:8000/diagram", timeout=30).json()
        diagram_url = diagram_response.get("diagram_url")
        if diagram_url:
            # Affichage via du HTML pour contourner le problème d'importation de PIL.imaging
            st.markdown(f"<img src='{diagram_url}' alt='Architecture fonctionnelle' style='width:100%;' />", unsafe_allow_html=True)
        else:
            st.error("Schéma non disponible")
    except Exception as e:
        st.error(f"Erreur lors de la récupération du schéma: {str(e)}")

# Démarrage du backend (en arrière-plan) si ce n'est pas déjà fait
if "backend_started" not in st.session_state:
    st.session_state.backend_started = True
    threading.Thread(target=run_backend, daemon=True).start()
    time.sleep(1)
