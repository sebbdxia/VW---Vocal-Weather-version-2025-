# weather.py
import requests
import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

def get_coordinates(city_name: str):
    geocode_url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "json"}
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(geocode_url, params=params, headers=headers)
    data = r.json()
    if not data:
        raise Exception(f"Ville introuvable : {city_name}")
    return float(data[0]["lat"]), float(data[0]["lon"])

def get_weather_forecast(city_name: str) -> pd.DataFrame:
    lat, lon = get_coordinates(city_name)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloudcover,windspeed_10m",
        "timezone": "auto"
    }
    response = retry_session.get(url, params=params)
    data = response.json()
    times = pd.to_datetime(data['hourly']['time'])
    df = pd.DataFrame({
        "date": times,
        "temperature_2m": data['hourly']['temperature_2m'],
        "cloudcover": data['hourly']['cloudcover'],
        "windspeed_10m": data['hourly']['windspeed_10m'],
        "pm2_5": [12.3] * len(times)
    })
    return df
