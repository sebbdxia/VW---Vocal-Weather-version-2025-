from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import azure.cognitiveservices.speech as speechsdk
import requests
import json
import os
from pydantic import BaseModel

# ✅ Configuration PostgreSQL (Azure)
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")  
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")  

# ✅ URL de connexion PostgreSQL
DATABASE_URL = f"postgresql://sebastien:GRETAP4!2025***@vw-sebastien.postgres.database.azure.com/postgres"

# ✅ Initialisation SQLAlchemy
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ✅ Vérification de la connexion PostgreSQL
try:
    conn = engine.connect()
    print("✅ Connexion à la base de données réussie !")
    conn.close()
except Exception as e:
    print(f"❌ Erreur de connexion à la base de données : {e}")

# 🎙️ Configuration Azure Speech-to-Text
AZURE_SPEECH_KEY = "54124b94ae904eeea1d8a652a4c3d88d"
AZURE_REGION = "francecentral"

# 🌦️ Configuration API Météo
WEATHER_API_KEY = "VOTRE_CLEF_METEO"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# 📌 Modèle SQLAlchemy pour stocker les prévisions météo
class WeatherRequest(Base):
    __tablename__ = "weather_requests"
    id = Column(Integer, primary_key=True, index=True)
    location = Column(String)
    date = Column(String)
    weather_data = Column(String)

# 📌 Création des tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

# ✅ Dépendance pour la gestion des sessions SQLAlchemy
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 📌 Modèle Pydantic pour la requête API
class WeatherAudio(BaseModel):
    audio_file: str

# ✅ Initialisation de l'application FastAPI
app = FastAPI()

# 🔹 Route pour analyser un fichier audio et récupérer la météo
@app.post("/process_weather/")
def process_weather(request: WeatherAudio, db: Session = Depends(get_db)):
    # 🎙️ Convertir l'audio en texte avec Azure Speech-to-Text
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    audio_config = speechsdk.AudioConfig(filename=request.audio_file)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = recognizer.recognize_once()
    
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise HTTPException(status_code=400, detail="Échec de la reconnaissance vocale")

    text = result.text
    location = text  # ⚠️ Supposition : l'audio contient directement le nom du lieu

    # 🌦️ Appel à l'API météo
    params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
    response = requests.get(WEATHER_API_URL, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Erreur API météo")

    weather_data = response.json()

    # 💾 Sauvegarde en base SQL
    db_request = WeatherRequest(location=location, date="today", weather_data=json.dumps(weather_data))
    db.add(db_request)
    db.commit()
    
    return {"location": location, "weather": weather_data}
