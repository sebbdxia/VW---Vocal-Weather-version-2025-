import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings, AudioProcessorBase
import numpy as np
import wave
import io
import requests
import pandas as pd
import plotly.express as px
import json

st.set_page_config(page_title="Météo & Analyse", layout="wide")
st.title("Prévisions Météo par Commande Vocale et Analyse de Performance")

# Définition d'un processeur audio pour enregistrer les frames
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []  # Liste qui stocke les tableaux numpy des frames audio

    def recv(self, frame):
        # Convertir le frame audio en tableau numpy et le stocker
        audio_frame = frame.to_ndarray()
        self.frames.append(audio_frame)
        return frame

# Démarrage du composant webrtc pour capter uniquement l'audio (sans vidéo)
webrtc_ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.RECVONLY,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_processor_factory=AudioRecorder,
)

# Création de deux onglets : "Commande Vocale" et "Analyse et Monitoring"
tab_commande, tab_analyse = st.tabs(["Commande Vocale", "Analyse et Monitoring"])

with tab_commande:
    st.header("Commande Vocale")
    st.write("L'enregistrement s'effectue en continu via le microphone. Lorsque vous souhaitez arrêter et envoyer votre commande, cliquez sur le bouton ci-dessous.")
    
    if st.button("Arrêter l'enregistrement et envoyer"):
        if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.frames:
            # Concaténation des frames enregistrées
            try:
                frames = np.concatenate(webrtc_ctx.audio_processor.frames, axis=0)
            except Exception as e:
                st.error("Erreur lors de la concaténation des frames : " + str(e))
                frames = None

            if frames is not None:
                # Conversion en fichier WAV dans un buffer mémoire
                buffer = io.BytesIO()
                try:
                    with wave.open(buffer, 'wb') as wf:
                        # Hypothèses : enregistrement mono, 16 bits, 44100 Hz
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(44100)
                        wf.writeframes(frames.tobytes())
                    wav_bytes = buffer.getvalue()
                except Exception as e:
                    st.error("Erreur lors de la création du fichier WAV : " + str(e))
                    wav_bytes = None

                if wav_bytes:
                    st.audio(wav_bytes, format="audio/wav")
                    # Préparer le fichier pour l'envoi vers le backend
                    files = {"file": ("recording.wav", io.BytesIO(wav_bytes), "audio/wav")}
                    headers = {"api_key": "secret-api-key"}  # Remplacez par votre clé réelle
                    try:
                        response = requests.post("http://localhost:8000/process", files=files, headers=headers)
                        if response.status_code == 200:
                            data = response.json()
                            st.success("Prévision reçue avec succès.")
                            st.markdown(f"**Lieu :** {data['location']}  \n**Horizon temporel :** {data['time_horizon']}")
                            st.write("### Détails de la prévision")
                            st.json(data["forecast"])
                            if "main" in data["forecast"]:
                                temp = data["forecast"]["main"].get("temp")
                                st.write(f"Température actuelle : {temp} °C")
                            # Affichage d'une carte interactive (exemple pour Paris)
                            if data["location"].lower() == "paris":
                                lat, lon = 48.8566, 2.3522
                            else:
                                lat, lon = 0, 0
                            df_map = pd.DataFrame({"lat": [lat], "lon": [lon]})
                            st.map(df_map)
                        else:
                            st.error("Erreur lors de l'appel à l'API : " + response.text)
                    except Exception as e:
                        st.error("Une exception est survenue lors de l'envoi : " + str(e))
        else:
            st.warning("Aucune donnée audio enregistrée. Veuillez vérifier que le microphone fonctionne correctement.")

with tab_analyse:
    st.header("Analyse et Monitoring")
    st.write("Cet espace permet de fournir un feedback sur les prévisions et d'observer des statistiques pour améliorer le système.")
    
    st.subheader("Feedback et Comparaison des Prévisions")
    st.write("Saisissez l'identifiant de la requête et la température réelle observée afin de comparer avec la prévision initiale.")
    
    query_id = st.text_input("Identifiant de la requête (query_id)", placeholder="Identifiant fourni lors de la commande")
    actual_temp = st.number_input("Température réelle observée (en °C)", value=0.0, step=0.1)
    
    if st.button("Envoyer mon Feedback"):
        feedback_payload = {
            "query_id": query_id,
            "actual_forecast": {"main": {"temp": actual_temp}}
        }
        headers = {"api_key": "secret-api-key", "Content-Type": "application/json"}
        try:
            response = requests.post("http://localhost:8000/feedback", data=json.dumps(feedback_payload), headers=headers)
            if response.status_code == 200:
                st.success("Feedback envoyé avec succès.")
                st.json(response.json())
            else:
                st.error("Erreur lors de l'envoi du feedback : " + response.text)
        except Exception as e:
            st.error("Une exception est survenue lors de l'envoi du feedback : " + str(e))
    
    st.subheader("Tableau de Bord Analytique")
    st.write("Les graphiques ci-dessous illustrent une simulation de l'évolution des prévisions par rapport aux valeurs réelles.")
    
    # Simulation de données analytiques pour la démonstration
    dates = pd.date_range(start="2025-01-01", periods=10, freq='D')
    data = {
        "Date": dates,
        "Température Prévue": np.random.normal(loc=20, scale=2, size=10),
        "Température Réelle": np.random.normal(loc=20, scale=3, size=10)
    }
    df = pd.DataFrame(data)
    df["Erreur Absolue"] = abs(df["Température Prévue"] - df["Température Réelle"])
    
    st.write("### Comparaison Température Prévue vs Réelle")
    fig_line = px.line(df, x="Date", y=["Température Prévue", "Température Réelle"], markers=True,
                       title="Évolution Temporelle des Températures")
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.write("### Erreur de Prédiction")
    fig_bar = px.bar(df, x="Date", y="Erreur Absolue", title="Erreur Absolue par Date")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.write("### Statistiques de Performance")
    moyenne_erreur = df["Erreur Absolue"].mean()
    ecart_type_erreur = df["Erreur Absolue"].std()
    st.markdown(f"**Moyenne de l'erreur absolue :** {moyenne_erreur:.2f} °C")
    st.markdown(f"**Écart-type de l'erreur :** {ecart_type_erreur:.2f} °C")
