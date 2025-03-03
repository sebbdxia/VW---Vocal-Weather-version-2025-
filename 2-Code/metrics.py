# metrics.py
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR

def init_metrics():
    registry = CollectorRegistry(auto_describe=True)
    REQUEST_COUNT = Counter("http_requests_total", "Nombre total de requêtes HTTP", ["method", "endpoint", "http_status"], registry=registry)
    REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Durée des requêtes HTTP (en secondes)", ["method", "endpoint"], registry=registry)
    FORECAST_REQUESTS = Counter("forecast_requests_total", "Nombre total de demandes de prévisions traitées", registry=registry)
    ERRORS_COUNT = Counter("errors_total", "Nombre total d'erreurs survenues", registry=registry)
    FEEDBACK_COUNT = Counter("feedback_total", "Nombre total de retours utilisateurs enregistrés", registry=registry)
    registry.register(PROCESS_COLLECTOR)
    registry.register(PLATFORM_COLLECTOR)
    registry.register(GC_COLLECTOR)
    return registry, REQUEST_COUNT, REQUEST_LATENCY, FORECAST_REQUESTS, ERRORS_COUNT, FEEDBACK_COUNT
