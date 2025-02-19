import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.title("Prévisions Météo avec Reconnaissance Vocale")

# Capture audio et envoi à l'API
st.header("1. Reconnaissance Vocale")
audio_file = st.file_uploader("Uploader un fichier audio", type=["wav", "mp3", "ogg"])
if audio_file:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.read())
    response = requests.post(f"{API_URL}/speech-to-text/", json={"audio_file": "temp_audio.wav"})
    if response.status_code == 200:
        text_output = response.json()["text"]
        st.write("Texte reconnu :", text_output)
    else:
        st.error("Erreur lors de la reconnaissance vocale")

# Traitement NLP
st.header("2. Analyse NLP")
user_text = st.text_area("Entrer un texte pour extraire les entités :")
if st.button("Analyser"):
    response = requests.post(f"{API_URL}/nlp/", json={"text": user_text})
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Erreur dans l'analyse NLP")

# Récupération des prévisions météo
st.header("3. Prévisions Météo")
location = st.text_input("Entrez une ville :")
forecast_horizon = st.slider("Horizon de prévision (en jours)", 1, 7, 3)
if st.button("Obtenir les prévisions"):
    response = requests.post(f"{API_URL}/weather/", json={"location": location, "forecast_horizon": forecast_horizon})
    if response.status_code == 200:
        st.json(response.json())
    else:
        st.error("Erreur lors de la récupération des prévisions météo")
