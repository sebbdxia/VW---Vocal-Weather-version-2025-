import os
import sys
import subprocess
import logging
import requests
import tarfile
import sounddevice as sd
import numpy as np
from pocketsphinx import Decoder
from dotenv import load_dotenv
import spacy
import uvicorn
import streamlit as st
import threading
import httpx
from fastapi import FastAPI, HTTPException

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Chargement des variables d'environnement ---
load_dotenv()
API_WEATHER_KEY = os.getenv("API_WEATHER_KEY")

if not API_WEATHER_KEY:
    logging.error("⚠️ La clé API OpenWeatherMap n'est pas définie. Ajoutez-la dans `.env`")
    sys.exit(1)

# --- Téléchargement et vérification du modèle ---
MODEL_FOLDER = os.path.abspath("model_fr")
TAR_URL = "https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/French/cmusphinx-fr-5.2.tar.gz/download"
TAR_PATH = os.path.join(MODEL_FOLDER, "cmusphinx-fr.tar.gz")

def download_file(url, dest_path):
    """Télécharge un fichier depuis une URL donnée."""
    logging.info(f"📥 Téléchargement de {os.path.basename(dest_path)}...")
    response = requests.get(url, stream=True)
    if response.status_code == 404:
        logging.error(f"❌ Fichier introuvable : {url}")
        sys.exit(1)

    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logging.info(f"✅ {os.path.basename(dest_path)} téléchargé avec succès.")

def extract_and_verify_model():
    """Télécharge et installe le modèle acoustique Pocketsphinx pour le français."""
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)

    if not os.path.exists(os.path.join(MODEL_FOLDER, "mdef")):  # Vérifier si le modèle est déjà extrait
        logging.info("📥 Téléchargement du modèle acoustique depuis SourceForge...")
        download_file(TAR_URL, TAR_PATH)

        logging.info("📦 Extraction du modèle...")
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=MODEL_FOLDER)

        extracted_folder = os.path.join(MODEL_FOLDER, "cmusphinx-fr-5.2")

        if os.path.exists(extracted_folder):
            for file in os.listdir(extracted_folder):
                shutil.move(os.path.join(extracted_folder, file), MODEL_FOLDER)
            shutil.rmtree(extracted_folder)

        os.remove(TAR_PATH)
        logging.info("✅ Modèle Pocketsphinx installé avec succès.")

extract_and_verify_model()

# --- Vérification des fichiers ---
def check_pocketsphinx_model():
    """Vérifie la présence des fichiers du modèle Pocketsphinx"""
    required_files = ["mdef", "means", "variances", "transition_matrices", "noisedict"]
    for f in required_files:
        path = os.path.join(MODEL_FOLDER, f)
        if not os.path.exists(path):
            logging.error(f"❌ Fichier manquant : {f}. Essayez de supprimer `model_fr` et relancer.")
            sys.exit(1)

check_pocketsphinx_model()

# --- Chargement du modèle spaCy ---
try:
    nlp = spacy.load("fr_core_news_sm")
    logging.info("✅ Modèle spaCy chargé avec succès.")
except OSError:
    logging.info("📥 Téléchargement du modèle spaCy...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
    nlp = spacy.load("fr_core_news_sm")

# --- Fonction de reconnaissance vocale ---
def recognize_speech_pocketsphinx(duration=5):
    """Enregistre et reconnaît la parole avec Pocketsphinx."""
    logging.info("🎤 Enregistrement audio en cours...")
    fs = 16000
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    audio_bytes = audio_data.tobytes()

    # Vérification et configuration de Pocketsphinx
    config = Decoder.default_config()
    config.set_string('-hmm', MODEL_FOLDER)
    config.set_string('-dict', os.path.join(MODEL_FOLDER, "noisedict"))

    try:
        decoder = Decoder(config)
    except RuntimeError as e:
        logging.error(f"❌ Erreur Pocketsphinx : {e}")
        sys.exit(1)

    decoder.start_utt()
    decoder.process_raw(audio_bytes, no_search=False, full_utt=True)
    decoder.end_utt()

    hypothesis = decoder.hyp()
    return hypothesis.hypstr if hypothesis else ""

# --- API Météo avec FastAPI ---
app = FastAPI()

@app.get("/meteo/{ville}")
async def meteo(ville: str):
    """Endpoint API pour obtenir la météo d'une ville."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={ville}&appid={API_WEATHER_KEY}&units=metric&lang=fr"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code == 200:
        data = response.json()
        return {
            "ville": ville,
            "description": data["weather"][0]["description"],
            "température": data["main"]["temp"],
            "humidité": data["main"]["humidity"],
            "vent": data["wind"]["speed"]
        }
    raise HTTPException(status_code=404, detail="Ville non trouvée.")

# --- Interface utilisateur avec Streamlit ---
def run_streamlit():
    """Interface utilisateur Streamlit."""
    st.title("Assistant Météo Vocal 🌦️")
    st.markdown("### Demandez la météo en parlant 🎙️")

    if st.button("🎤 Parler"):
        text = recognize_speech_pocketsphinx(duration=5)
        st.write(f"Texte reconnu : {text}")

# --- Lancement de l'application ---
if __name__ == "__main__":
    api_thread = threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000), daemon=True)
    api_thread.start()
    run_streamlit()
