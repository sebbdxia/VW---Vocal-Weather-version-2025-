import streamlit as st
import requests
import plotly.express as px
import pandas as pd

BACKEND_URL = "http://127.0.0.1:8000"

st.title("🌤️ Application Météo Vocale")

file = st.file_uploader("🎤 Chargez un fichier audio", type=["wav", "mp3"])
if file:
    with st.spinner("Analyse de l'audio..."):
        response = requests.post(f"{BACKEND_URL}/recognize_speech", files={"file": file})
        if response.status_code == 200:
            text = response.json()["text"]
            st.write("🗣️ Texte reconnu :", text)

            intent_response = requests.post(f"{BACKEND_URL}/extract_intent", json={"text": text})
            if intent_response.status_code == 200:
                intent_data = intent_response.json()
                city, date = intent_data["city"], intent_data["date"]
                st.write(f"📍 Ville : {city}, 📅 Date : {date}")

                weather_response = requests.get(f"{BACKEND_URL}/get_weather/{city}/{date}")
                if weather_response.status_code == 200:
                    weather_data = weather_response.json()
                    st.write("🌡️ Température :", weather_data["main"]["temp"])
                    st.write("🌦️ Conditions :", weather_data["weather"][0]["description"])

                    df = pd.DataFrame({"Température": [weather_data["main"]["temp"]], "Humidité": [weather_data["main"]["humidity"]]})
                    fig = px.bar(df, x=df.columns, y=df.iloc[0], title="Conditions Météo")
                    st.plotly_chart(fig)
                else:
                    st.error("❌ Impossible de récupérer les prévisions météo")
            else:
                st.error("❌ Extraction des intentions échouée")
        else:
            st.error("❌ Échec de la reconnaissance vocale")
