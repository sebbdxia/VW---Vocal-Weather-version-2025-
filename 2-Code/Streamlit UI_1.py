import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue

# Définition d'un processeur audio pour capturer les frames audio
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.put(frame)
        return frame

    def get_audio_bytes(self):
        frames = []
        while not self.audio_frames.empty():
            frame = self.audio_frames.get()
            # Convertir le frame en tableau numpy puis en bytes
            frames.append(frame.to_ndarray().tobytes())
        return b"".join(frames)

st.title("Application Météo – Commande vocale / manuelle")
st.write("Obtenez la prévision météo en utilisant une commande vocale (enregistrement via micro ou fichier audio) ou en saisissant manuellement.")

# Choix du mode de commande
mode = st.radio("Choisissez votre mode de commande :", ("Enregistrement par micro", "Charger un fichier audio", "Manuelle"))

audio_bytes = None
transcription_input = ""
city_input = ""
horizon_input = ""

if mode == "Enregistrement par micro":
    st.subheader("Enregistrement par micro")
    st.write("Cliquez sur le bouton pour démarrer l'enregistrement via votre micro.")
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        client_settings={
            "RTCConfiguration": {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            "mediaStreamConstraints": {"audio": True, "video": False},
        },
        audio_processor_factory=AudioRecorder
    )
    if webrtc_ctx.audio_processor:
        if st.button("Arrêter l'enregistrement"):
            audio_bytes = webrtc_ctx.audio_processor.get_audio_bytes()
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.info("Enregistrement capturé.")
            else:
                st.warning("Aucune donnée audio capturée.")
    st.write("Vous pouvez compléter ou modifier la transcription obtenue :")
    transcription_input = st.text_input("Transcription (optionnelle)")
    st.write("Vous pouvez également saisir manuellement la ville et l’horizon temporel :")
    city_input = st.text_input("Ville (optionnel)")
    horizon_input = st.text_input("Horizon temporel (optionnel)")

elif mode == "Charger un fichier audio":
    st.subheader("Charger un fichier audio")
    st.write("Chargez un fichier audio (format WAV ou MP3).")
    uploaded_file = st.file_uploader("Sélectionnez votre fichier audio", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_bytes = uploaded_file.getvalue()
        st.audio(audio_bytes, format="audio/wav")
    st.write("Vous pouvez compléter ou modifier la transcription obtenue :")
    transcription_input = st.text_input("Transcription (optionnelle)")
    st.write("Vous pouvez également saisir manuellement la ville et l’horizon temporel :")
    city_input = st.text_input("Ville (optionnel)")
    horizon_input = st.text_input("Horizon temporel (optionnel)")
    
else:  # Mode manuel
    st.subheader("Commande manuelle")
    transcription_input = st.text_input("Transcription de la commande (facultatif)")
    city_input = st.text_input("Ville")
    horizon_input = st.text_input("Horizon temporel (ex. demain, aujourd'hui, etc.)")

if st.button("Envoyer la commande"):
    backend_url = "http://localhost:8000/process_command"
    files = {}
    data = {}

    if audio_bytes is not None:
        # On envoie le fichier audio sous forme d'un tuple (nom, bytes, type MIME)
        files["file"] = ("audio.wav", audio_bytes, "audio/wav")
    if transcription_input:
        data["transcription"] = transcription_input
    if city_input:
        data["city"] = city_input
    if horizon_input:
        data["horizon"] = horizon_input

    try:
        response = requests.post(backend_url, files=files, data=data)
        if response.status_code == 200:
            result = response.json()
            st.success("Prévision pour " + result["location"])
            st.json(result["forecast"])
            if result.get("transcription"):
                st.write("Transcription utilisée :", result["transcription"])
        else:
            st.error("Une erreur est survenue lors de la récupération de la prévision.")
    except Exception as e:
        st.error("Impossible de joindre le backend. Vérifiez qu'il est démarré et accessible.")
