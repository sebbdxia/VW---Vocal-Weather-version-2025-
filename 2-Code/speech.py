# speech.py
import os
import tempfile
import logging
from config import SPEECH_KEY, SPEECH_REGION
import azure.cognitiveservices.speech as speechsdk
import time
import streamlit as st

def azure_speech_to_text(audio_bytes: bytes) -> str:
    if not SPEECH_KEY or not SPEECH_REGION:
        logging.error("Clés Azure Speech non configurées.")
        return ""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_filename = temp_audio.name
    audio_config = speechsdk.audio.AudioConfig(filename=temp_filename)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once_async().get()
    os.remove(temp_filename)
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        logging.error(f"Transcription annulée: {details.reason}. Détails: {details.error_details}")
        return ""
    else:
        logging.error(f"Erreur de transcription: {result.reason}")
        return ""

def azure_speech_from_microphone() -> str:
    if not SPEECH_KEY or not SPEECH_REGION:
        logging.error("Les clés d'API Azure Speech ne sont pas configurées.")
        return ""
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "fr-FR"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    st.info("Veuillez parler... (enregistrement continu pendant 10 secondes)")
    result_texts = []
    def recognized_cb(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            result_texts.append(evt.result.text)
    def stop_cb(evt):
        st.info("Fin de l'enregistrement.")
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)
    speech_recognizer.start_continuous_recognition()
    time.sleep(10)
    speech_recognizer.stop_continuous_recognition()
    final_text = " ".join(result_texts)
    if not final_text:
        logging.error("Aucune parole détectée lors de l'enregistrement continu.")
    return final_text
