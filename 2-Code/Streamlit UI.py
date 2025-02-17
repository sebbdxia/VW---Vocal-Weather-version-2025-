import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("🌦 Vocal Weather 2025")

# Gestion des utilisateurs
username = st.text_input("Nom d'utilisateur")
password = st.text_input("Mot de passe", type="password")
if st.button("S'inscrire"):
    response = requests.post(f"{API_URL}/register/", json={"username": username, "password": password})
    if response.status_code == 200:
        st.success("Inscription réussie !")
    else:
        st.error(response.json()["detail"])

if st.button("Se connecter"):
    response = requests.post(f"{API_URL}/login/", json={"username": username, "password": password})
    if response.status_code == 200:
        token = response.json()["access_token"]
        st.session_state["token"] = token
        st.success("Connecté !")
    else:
        st.error(response.json()["detail"])

# Upload Audio
audio_file = st.file_uploader("🎤 Téléchargez un fichier audio", type=["wav"])
if st.button("Analyser la météo 🎙️"):
    if "token" not in st.session_state:
        st.error("Vous devez être connecté pour utiliser cette fonctionnalité")
    else:
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        response = requests.post(f"{API_URL}/process_weather/", json={"audio_file": "temp_audio.wav"}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            st.write(f"📍 Lieu : {data['location']}")
            st.write(f"🌡 Température : {data['weather']['main']['temp']}°C")
        else:
            st.error(response.json()["detail"])
