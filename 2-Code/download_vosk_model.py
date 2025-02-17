import os
import sys
import urllib.request
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

vosk_model_url = "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"
vosk_model_path = "model_fr"
vosk_model_zip = "vosk-model-small-fr-0.22.zip"

if not os.path.exists(vosk_model_path):
    logging.info("Le dossier du modèle Vosk '%s' est introuvable. Téléchargement en cours...", vosk_model_path)
    try:
        urllib.request.urlretrieve(vosk_model_url, vosk_model_zip)
        with zipfile.ZipFile(vosk_model_zip, 'r') as zip_ref:
            zip_ref.extractall(vosk_model_path)
        os.remove(vosk_model_zip)
        logging.info("Le modèle Vosk a été téléchargé et extrait avec succès.")
    except Exception as e:
        logging.error("Erreur lors du téléchargement ou de l'extraction du modèle Vosk : %s", e)
        sys.exit(1)
else:
    logging.info("Le dossier du modèle Vosk '%s' existe déjà.", vosk_model_path)
