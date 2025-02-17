from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
import datetime
import azure.cognitiveservices.speech as speechsdk
import requests
import json
from pydantic import BaseModel

# Configuration SQL (Azure SQL Database)
DATABASE_URL = "mssql+pyodbc://user:password@server.database.windows.net/dbname?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Clé secrète pour JWT
SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"

# Configuration Azure
AZURE_SPEECH_KEY = "VOTRE_CLEF_AZURE_SPEECH"
AZURE_REGION = "VOTRE_REGION"
WEATHER_API_KEY = "VOTRE_CLEF_METEO"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"

# Sécurité des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Modèles SQL
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class WeatherRequest(Base):
    __tablename__ = "weather_requests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    location = Column(String)
    date = Column(String)
    weather_data = Column(String)

Base.metadata.create_all(bind=engine)

# Modèles Pydantic
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class WeatherAudio(BaseModel):
    audio_file: str

# Création d'un utilisateur
def create_user(db: Session, user: UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Vérification du mot de passe
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Authentification utilisateur
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

# Génération de JWT
def create_jwt_token(data: dict):
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# Middleware pour récupérer l'utilisateur
def get_current_user(token: str, db: Session = Depends(SessionLocal)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(User).filter(User.username == payload.get("sub")).first()
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

# FastAPI App
app = FastAPI()

@app.post("/register/", response_model=Token)
def register(user: UserCreate, db: Session = Depends(SessionLocal)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Nom d'utilisateur déjà pris")
    user = create_user(db, user)
    token = create_jwt_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login/", response_model=Token)
def login(user: UserLogin, db: Session = Depends(SessionLocal)):
    user = authenticate_user(db, user.username, user.password)
    if not user:
        raise HTTPException(status_code=400, detail="Identifiants incorrects")
    token = create_jwt_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/process_weather/")
def process_weather(request: WeatherAudio, current_user: User = Depends(get_current_user), db: Session = Depends(SessionLocal)):
    # Convertir l'audio en texte avec Azure Speech-to-Text
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    audio_config = speechsdk.AudioConfig(filename=request.audio_file)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = recognizer.recognize_once()
    
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise HTTPException(status_code=400, detail="Échec de la reconnaissance vocale")

    text = result.text
    location = text  # Supposons que le texte soit directement le lieu

    # Récupérer la météo
    params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
    response = requests.get(WEATHER_API_URL, params=params)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Erreur API météo")

    weather_data = response.json()

    # Sauvegarde en base SQL
    db_request = WeatherRequest(user_id=current_user.id, location=location, date="today", weather_data=json.dumps(weather_data))
    db.add(db_request)
    db.commit()
    
    return {"location": location, "weather": weather_data}
