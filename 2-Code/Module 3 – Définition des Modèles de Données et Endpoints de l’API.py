class WeatherResponse(BaseModel):
    location: str
    forecast: dict
    forecast_days: int
    message: str = None
    transcription: str = None
    mode: str = None

# Endpoint principal qui traite la commande (fichier audio ou saisie manuelle) et renvoie la prévision météo.
@app.post("/process_command", response_model=WeatherResponse)
@measure_latency
async def process_command(
    file: UploadFile = File(None),
    transcription: str = Form(None),
    city: str = Form(None),
    forecast_days: str = Form(None)
):
    try:
        if file is not None:
            audio_bytes = await file.read()
            transcription_result = azure_speech_to_text(audio_bytes)
            transcription_final = transcription_result
        else:
            transcription_final = transcription or ""
        
        extracted_city, extracted_days = spacy_analyze(transcription_final)
        logging.info(f"Extraction: Ville='{extracted_city}', Jours='{extracted_days}'")
        final_city = city if city else extracted_city
        
        if (file is not None) or (transcription_final and not forecast_days):
            final_forecast_days = extracted_days
        else:
            final_forecast_days = int(forecast_days) if forecast_days is not None else 7
        
        hourly_dataframe = get_weather_forecast(final_city)
        logging.info(f"Prévision météo récupérée pour {final_city}")
        store_forecast_in_db(transcription_final, final_city, final_forecast_days, hourly_dataframe, "vocale" if file is not None else "manuel")
        
        FORECAST_REQUESTS.inc()
        forecast_dict = hourly_dataframe.to_dict(orient="records")
        return WeatherResponse(
            location=final_city,
            forecast={"hourly": forecast_dict},
            forecast_days=final_forecast_days,
            message="Prévision obtenue avec succès.",
            transcription=transcription_final,
            mode="vocale" if file is not None else "manuel"
        )
    except Exception as e:
        ERRORS_COUNT.inc()
        logging.error("Erreur lors du traitement de la commande", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la commande")

# Endpoints additionnels pour fournir des informations d’analyse, exposer les métriques et recueillir les retours.
@app.get("/analysis")
def analysis():
    return {"total_requests": len(logs), "logs": logs, "feedbacks": user_feedbacks}

@app.get("/metrics")
def metrics():
    try:
        data = generate_latest(prom_registry)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logging.error("Erreur lors de l'exposition des métriques", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de l'exposition des métriques")

@app.get("/top_cities")
def top_cities():
    city_counts = {}
    for entry in logs:
        city = entry.get("city", "Inconnu")
        city_counts[city] = city_counts.get(city, 0) + 1
    return city_counts

@app.get("/feedbacks")
def get_feedbacks():
    return user_feedbacks

@app.post("/feedback")
def feedback(rating: int = Form(...), comment: str = Form("")):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rating": rating,
        "comment": comment
    }
    user_feedbacks.append(entry)
    FEEDBACK_COUNT.inc()
    try:
        import psycopg2
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ,
                rating INTEGER,
                comment TEXT
            );
        """)
        cur.execute("""
            INSERT INTO feedback (timestamp, rating, comment)
            VALUES (%s, %s, %s)
        """, (entry["timestamp"], entry["rating"], entry["comment"]))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du stockage du feedback PostgreSQL: {e}")
    return {"status": "Feedback enregistré"}
