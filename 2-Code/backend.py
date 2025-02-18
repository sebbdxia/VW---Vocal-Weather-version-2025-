from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import azure.cognitiveservices.speech as speechsdk
import requests
import json
import os
from pydantic import BaseModel, Field
import logging
import tempfile

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration PostgreSQL (Azure)
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DATABASE_URL = f"postgresql://sebastien:GRETAP4!2025***@vw-sebastien.postgres.database.azure.com/postgres"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Modèles SQLAlchemy pour stocker les prévisions et les feedbacks
class WeatherRequest(Base):
    __tablename__ = "weather_requests"
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String)
    time_horizon = Column(String)
    weather_data = Column(Text)

class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String, index=True)
    actual_forecast = Column(Text)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Configuration Azure Speech-to-Text
AZURE_SPEECH_KEY = "54124b94ae904eeea1d8a652a4c3d88d"
AZURE_REGION = "francecentral"

# Configuration API Météo
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "VOTRE_CLEF_METEO")
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Modèle Pydantic pour le feedback
class FeedbackPayload(BaseModel):
    query_id: str = Field(..., example="12345")
    actual_forecast: dict = Field(..., example={"main": {"temp": 22.5}})

# Fonction simulant le traitement NLP pour extraire la localité et l'horizon depuis une transcription
def extract_entities(text: str):
    try:
        parts = text.split(',')
        location = parts[0].strip() if parts else text.strip()
        time_horizon = parts[1].strip() if len(parts) > 1 else "today"
    except Exception as e:
        logger.error(f"Erreur d'extraction NLP: {e}")
        location, time_horizon = text.strip(), "today"
    return location, time_horizon

app = FastAPI()

@app.post("/process")
async def process_weather(
    file: UploadFile = File(None),
    manual_location: str = Form(None),
    manual_date: str = Form(None),
    db: Session = Depends(get_db)
):
    transcription = ""
    # Si un fichier audio est fourni, on réalise la reconnaissance vocale
    if file is not None:
        try:
            contents = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(contents)
                temp_file_path = tmp.name
        except Exception as e:
            logger.error(f"Erreur lors de la création du fichier temporaire: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la création du fichier audio temporaire")
        
        try:
            speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
            audio_config = speechsdk.AudioConfig(filename=temp_file_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
            result = recognizer.recognize_once()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcription = result.text
            else:
                logger.error("Échec de la reconnaissance vocale")
                raise HTTPException(status_code=400, detail="Échec de la reconnaissance vocale")
        except Exception as e:
            logger.error(f"Erreur lors de la reconnaissance vocale: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la reconnaissance vocale")
        finally:
            os.remove(temp_file_path)
    
    # Détermination de la localité : la saisie manuelle prime sur la commande vocale
    if manual_location:
        location = manual_location
    elif transcription:
        location, extracted_time_horizon = extract_entities(transcription)
    else:
        raise HTTPException(status_code=400, detail="Aucune information fournie pour la localité (commande vocale ou saisie manuelle requise).")
    
    # Détermination de l'horizon temporel : la saisie manuelle prime, sinon on extrait ou on utilise une valeur par défaut
    if manual_date:
        time_horizon = manual_date
    elif transcription:
        # Si aucune date manuelle n'est fournie, on tente d'extraire l'horizon depuis la transcription
        if not manual_location:
            _, extracted_time_horizon = extract_entities(transcription)
            time_horizon = extracted_time_horizon
        else:
            time_horizon = "today"
    else:
        time_horizon = "today"
    
    logger.info(f"Localité utilisée: {location} | Horizon temporel: {time_horizon} | Transcription: {transcription}")
    
    # Appel à l'API météo
    params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'API météo: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'appel à l'API météo")
    
    weather_data = response.json()

    try:
        db_request = WeatherRequest(location=location, time_horizon=time_horizon, weather_data=json.dumps(weather_data))
        db.add(db_request)
        db.commit()
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement en base: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement des données")
    
    return {
        "transcription": transcription,
        "location": location,
        "time_horizon": time_horizon,
        "forecast": weather_data
    }

@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload, db: Session = Depends(get_db)):
    try:
        feedback_entry = Feedback(query_id=payload.query_id, actual_forecast=json.dumps(payload.actual_forecast))
        db.add(feedback_entry)
        db.commit()
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement du feedback")
    
    return {"message": "Feedback enregistré avec succès.", "query_id": payload.query_id}
