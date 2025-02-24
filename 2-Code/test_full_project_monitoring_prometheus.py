import pytest
from fastapi.testclient import TestClient
import importlib.util

# Chargement dynamique du module source depuis le chemin fourni
spec = importlib.util.spec_from_file_location(
    "full_project_module", 
    r"C:\Users\sbond\Desktop\VW - Vocal Weather (version 2025)\2-Code\full_ porject_monitorig_prometheus.py"
)
full_project_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_project_module)

# Récupération de l'instance FastAPI depuis le module importé
app = full_project_module.app

client = TestClient(app)

def test_analysis_endpoint():
    """
    Vérifie que l'endpoint /analysis renvoie bien les clés attendues.
    """
    response = client.get("/analysis")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data, "La réponse doit contenir la clé 'total_requests'"
    assert "logs" in data, "La réponse doit contenir la clé 'logs'"
    assert "feedbacks" in data, "La réponse doit contenir la clé 'feedbacks'"

def test_metrics_endpoint():
    """
    Vérifie que l'endpoint /metrics retourne les métriques Prometheus attendues.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    # Le content-type devrait inclure 'text/plain'
    assert "text/plain" in response.headers.get("content-type", ""), "Le content-type doit être text/plain"
    assert "http_requests_total" in response.text, "La métrique http_requests_total doit être présente dans la réponse"

def test_feedback_submission_and_retrieval():
    """
    Soumet un feedback et vérifie qu'il est présent dans la liste des feedbacks.
    """
    feedback_data = {"rating": 4, "comment": "Bon service"}
    post_response = client.post("/feedback", data=feedback_data)
    assert post_response.status_code == 200, "La soumission du feedback doit réussir"
    
    get_response = client.get("/feedbacks")
    assert get_response.status_code == 200, "La récupération des feedbacks doit réussir"
    feedbacks = get_response.json()
    # Vérifie qu'au moins un feedback possède la note soumise
    assert any(fb.get("rating") == 4 for fb in feedbacks), "Le feedback soumis doit être présent dans la liste"

def test_process_command_with_transcription():
    """
    Teste l'endpoint /process_command en fournissant une transcription simple.
    """
    payload = {
        "transcription": "Prévision pour Paris sur 3 jours",
        "city": "Paris",
        "forecast_days": "3"
    }
    response = client.post("/process_command", data=payload)
    assert response.status_code == 200, "La commande doit être traitée sans erreur"
    data = response.json()
    assert "location" in data, "La réponse doit contenir la clé 'location'"
    assert data["location"] == "Paris", "La ville extraite doit être 'Paris'"
    assert "forecast" in data, "La réponse doit inclure les prévisions météo"
    assert "hourly" in data["forecast"], "Les prévisions horaires doivent être présentes"
