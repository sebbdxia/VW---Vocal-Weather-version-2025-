import os
import requests
import azure.cognitiveservices.speech as speechsdk
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from jose import JWTError, jwt
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration sécurisée
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
DB_URL = os.getenv("DB_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

# Initialisation FastAPI
app = FastAPI()

# Connexion à la base de données Azure SQL
engine = sqlalchemy.create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Vérification du token JWT
def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

# Reconnaissance vocale avec Azure Speech-to-Text
@app.post("/recognize_speech")
async def recognize_speech(file: UploadFile = File(...)):
    try:
        file_path = f"temp_audio_{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        audio_input = speechsdk.audio.AudioConfig(filename=file_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
        result = speech_recognizer.recognize_once()

        os.remove(file_path)  # Nettoyage du fichier temporaire

        if not result.text:
            raise HTTPException(status_code=400, detail="Échec de la reconnaissance vocale")

        return {"text": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Extraction des intentions utilisateur (lieu et date)
def extract_intent(text: str):
    words = text.lower().split()
    city = "Paris" if "paris" in words else ("Londres" if "londres" in words else None)
    date = "tomorrow" if "demain" in words else "today"

    if not city:
        raise HTTPException(status_code=400, detail="Ville non reconnue dans le texte")

    return {"city": city, "date": date}

@app.post("/extract_intent")
def extract_user_intent(request: BaseModel):
    return extract_intent(request.text)

# API météo
@app.get("/get_weather/{city}/{date}")
def get_weather(city: str, date: str):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Ville non trouvée")

    return response.json()
