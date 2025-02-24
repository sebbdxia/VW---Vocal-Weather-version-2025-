app = FastAPI()
logging.basicConfig(level=logging.INFO)

nlp = spacy.load("fr_core_news_sm")
logs = []            
user_feedbacks = []

# Middleware qui intercepte chaque requête pour mesurer son temps de traitement et incrémenter les compteurs appropriés.
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed_time = time.time() - start_time
    path = request.url.path
    method = request.method
    status = response.status_code
    REQUEST_COUNT.labels(method=method, endpoint=path, http_status=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed_time)
    return response

# Décorateur complémentaire pour mesurer la latence de fonctions spécifiques.
def measure_latency(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start
        # La mesure complémentaire peut être étendue si nécessaire.
        return result
    return wrapper
