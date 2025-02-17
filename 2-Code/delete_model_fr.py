import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

model_folder = "model_fr"

if os.path.exists(model_folder):
    logging.info("Suppression du dossier '%s'...", model_folder)
    try:
        shutil.rmtree(model_folder)
        logging.info("Le dossier '%s' a été supprimé avec succès.", model_folder)
    except Exception as e:
        logging.error("Erreur lors de la suppression du dossier '%s' : %s", model_folder, e)
else:
    logging.info("Le dossier '%s' n'existe pas.", model_folder)
