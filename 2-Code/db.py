# db.py
import psycopg2
import json
import datetime
import logging
from config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME

def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST)

def create_forecasts_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ,
            transcription TEXT,
            city TEXT,
            forecast_days INTEGER,
            forecast JSONB,
            mode TEXT
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def store_forecast(transcription, city, forecast_days, forecast_data, mode):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "transcription": transcription,
        "city": city,
        "forecast_days": forecast_days,
        "forecast": forecast_data,
        "mode": mode
    }
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        create_forecasts_table()  # S'assurer que la table existe
        cur.execute("""
            INSERT INTO forecasts (timestamp, transcription, city, forecast_days, forecast, mode)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            entry["timestamp"],
            entry["transcription"],
            entry["city"],
            entry["forecast_days"],
            json.dumps(entry["forecast"], default=str),
            entry["mode"]
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f"Erreur lors du stockage en base de données : {e}")

def get_all_forecasts():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT timestamp, transcription, city, forecast_days, forecast, mode 
            FROM forecasts
            ORDER BY timestamp DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        logs_db = []
        for row in rows:
            logs_db.append({
                "timestamp": row[0],
                "transcription": row[1],
                "city": row[2],
                "forecast_days": row[3],
                "forecast": row[4],
                "mode": row[5]
            })
        return logs_db
    except Exception as e:
        logging.error("Erreur lors de la récupération des logs depuis Azure", exc_info=True)
        raise

def get_top_cities():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT city, COUNT(*) FROM forecasts GROUP BY city")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        logging.error("Erreur lors de la récupération des données de villes depuis Azure", exc_info=True)
        raise
