import queue
import json
import logging
import sounddevice as sd
from vosk import Model, KaldiRecognizer

def recognize_speech_vosk():
    try:
        # Chargement du modèle français (assurez-vous de télécharger le modèle et de préciser le chemin)
        model = Model("model_fr")
    except Exception as e:
        logging.error("Erreur lors du chargement du modèle Vosk : %s", e)
        return None

    recognizer = KaldiRecognizer(model, 16000)
    q = queue.Queue()

    def audio_callback(indata, frames, time, status):
        if status:
            logging.error("Erreur dans le flux audio : %s", status)
        q.put(bytes(indata))

    # Capture audio en temps réel avec une configuration adaptée à Vosk
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=audio_callback):
        logging.info("Parlez... (appuyez sur Ctrl+C pour arrêter)")
        while True:
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_dict = json.loads(result)
                recognized_text = result_dict.get("text", "")
                if recognized_text:
                    logging.info("Texte reconnu : %s", recognized_text)
                    return recognized_text
