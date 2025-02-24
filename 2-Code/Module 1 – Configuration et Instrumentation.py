#!/usr/bin/env python
import os
import json
import datetime
import logging
import spacy
import requests
import uvicorn
import threading
import time
import re
import tempfile
from typing import Tuple, Callable
from functools import wraps

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import streamlit as st
import requests_cache
import pandas as pd
from retry_requests import retry

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dotenv import load_dotenv
load_dotenv('.env')

# Configuration de la base de données à l’aide des variables d’environnement.
DB_USER = os.getenv("DB_USER", "sebastien")
DB_PASSWORD = os.getenv("DB_PASSWORD", "GRETAP4!2025***")
DB_HOST = os.getenv("DB_HOST", "vw-sebastien.postgres.database.azure.com")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Mise en place du registre Prometheus et des métriques afin de mesurer l’activité et la performance.
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
prom_registry = CollectorRegistry(auto_describe=True)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Nombre total de requêtes HTTP",
    ["method", "endpoint", "http_status"],
    registry=prom_registry
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP (en secondes)",
    ["method", "endpoint"],
    registry=prom_registry
)

FORECAST_REQUESTS = Counter(
    "forecast_requests_total",
    "Nombre total de demandes de prévisions traitées",
    registry=prom_registry
)
ERRORS_COUNT = Counter(
    "errors_total",
    "Nombre total d'erreurs survenues",
    registry=prom_registry
)
FEEDBACK_COUNT = Counter(
    "feedback_total",
    "Nombre total de retours utilisateurs enregistrés",
    registry=prom_registry
)

# Mise en cache des requêtes HTTP et configuration de la reprise automatique en cas d’échec.
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# Récupération des clés d’accès pour le service de transcription vocale via Azure.
SPEECH_KEY = os.environ.get("SPEECH_KEY")
SPEECH_REGION = os.environ.get("SPEECH_REGION")
