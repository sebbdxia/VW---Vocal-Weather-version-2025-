# config.py
import os
from dotenv import load_dotenv
import os.path

load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")
