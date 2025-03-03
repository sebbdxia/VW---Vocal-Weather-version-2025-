# streamlit_app.py
import time
import threading
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from weather import get_coordinates

def run_streamlit():
    st.title("Application Météo – Commande vocale et manuelle (Open-Meteo)")
    if "forecast_response" not in st.session_state:
        st.session_state["forecast_response"] = None
    tab1, tab2, tab3 = st.tabs(["Prévisions", "Analyse & Monitoring", "Feedback"])
    with tab1:
        st.header("Bienvenue sur l'application météo")
        mode = st.radio("Sélectionnez le mode de commande :", ("Enregistrement par micro", "Manuelle"))
        transcription_input = ""
        city_input = ""
        forecast_days_input = None
        if mode == "Enregistrement par micro":
            st.subheader("Commande vocale via microphone")
            if st.button("Enregistrer la commande vocale"):
                from speech import azure_speech_from_microphone
                transcription_result = azure_speech_from_microphone()
                if transcription_result:
                    st.session_state.micro_transcription = transcription_result
                else:
                    st.error("Enregistrement échoué ou aucune parole détectée. Veuillez réessayer.")
            if "micro_transcription" in st.session_state and st.session_state.micro_transcription:
                st.write("Transcription obtenue :", st.session_state.micro_transcription)
                city_input = st.text_input("Indiquez la ville (facultatif)")
                if st.button("Envoyer la commande vocale"):
                    backend_url = "http://localhost:8000/process_command"
                    data = {"transcription": st.session_state.micro_transcription, "mode": "vocale"}
                    if city_input:
                        data["city"] = city_input
                    response = requests.post(backend_url, data=data)
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.forecast_response = result
                        st.success(f"Prévision pour {result['location']} (commande vocale)")
                    else:
                        st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande vocale")
        else:
            st.subheader("Commande manuelle")
            city_input = st.text_input("Ville")
            forecast_days_input = st.selectbox("Nombre de jours de prévision", options=[3, 5, 7], index=2)
            transcription_input = st.text_input("Saisissez votre commande (ex : prévisions pour 3 jours)")
            if st.button("Envoyer la commande"):
                backend_url = "http://localhost:8000/process_command"
                data = {}
                if transcription_input:
                    data["transcription"] = transcription_input
                if city_input:
                    data["city"] = city_input
                if forecast_days_input is not None:
                    data["forecast_days"] = str(forecast_days_input)
                data["mode"] = "manuel"
                try:
                    response = requests.post(backend_url, data=data)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prévision pour {result['location']}")
                        st.session_state.forecast_response = result
                    else:
                        st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande")
                except Exception as e:
                    st.error("Impossible de joindre le backend. Vérifiez qu'il est démarré et accessible.")
        if st.session_state.forecast_response and st.button("Afficher les résultats"):
            result = st.session_state.forecast_response
            st.subheader("Prévisions")
            final_days = result["forecast_days"]
            df = pd.DataFrame(result["forecast"]["hourly"])
            df['date'] = pd.to_datetime(df['date'])
            df['hour'] = df['date'].dt.hour
            df_filtered = df[df['hour'] == 12].sort_values(by='date').head(final_days)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=("Température (°C)", "Nébulosité (%)", "Vent (km/h)"))
            fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['temperature_2m'],
                                     mode='lines+markers', marker=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['cloudcover'],
                                     mode='lines+markers', marker=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['windspeed_10m'],
                                     mode='lines+markers', marker=dict(color='green')), row=3, col=1)
            fig.update_layout(height=600, title=f"Prévisions de Midi sur {final_days} jours", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Détails des prévisions")
            st.dataframe(df_filtered[['date', 'temperature_2m', 'cloudcover', 'windspeed_10m', 'pm2_5']].rename(
                columns={
                    "date": "Date",
                    "temperature_2m": "Température (°C)",
                    "cloudcover": "Nébulosité (%)",
                    "windspeed_10m": "Vent (km/h)",
                    "pm2_5": "Pollution (µg/m³)"
                }
            ))
            st.subheader("Localisation")
            try:
                lat, lon = get_coordinates(result["location"])
                map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
                st.map(map_data)
            except Exception as e:
                st.error(f"Impossible d'afficher la carte: {e}")
            if result.get("transcription"):
                st.write("Transcription utilisée :", result["transcription"])
    with tab2:
        st.header("Analyse et Monitoring")
        try:
            analysis_url = "http://localhost:8000/analysis"
            response = requests.get(analysis_url)
            if response.status_code == 200:
                data_analysis = response.json()
                st.write("Nombre total de requêtes :", data_analysis.get("total_requests", 0))
                st.write("Nombre de retours utilisateurs :", len(data_analysis.get("feedbacks", [])))
            else:
                st.error("Erreur lors de la récupération des données d'analyse.")
        except Exception as e:
            st.error("Impossible de joindre le backend pour l'analyse.")
        try:
            metrics_url = "http://localhost:8000/metrics"
            metrics_response = requests.get(metrics_url)
            if metrics_response.status_code == 200:
                lines = metrics_response.text.splitlines()
                metrics_list = []
                for line in lines:
                    if line.startswith("#") or line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = parts[1]
                        metrics_list.append({"Metric": metric_name, "Value": metric_value})
                df_metrics = pd.DataFrame(metrics_list)
                st.subheader("Métriques Prometheus")
                st.dataframe(df_metrics)
            else:
                st.error("Erreur lors de la récupération des métriques Prometheus.")
        except Exception as e:
            st.error("Impossible de joindre l'endpoint /metrics.")
        try:
            top_cities_url = "http://localhost:8000/top_cities"
            response = requests.get(top_cities_url)
            if response.status_code == 200:
                st.subheader("Répartition des demandes par ville")
                top_cities_data = response.json()
                df_top = pd.DataFrame(list(top_cities_data.items()), columns=["Ville", "Nombre de demandes"])
                st.dataframe(df_top)
                fig = go.Figure(data=[go.Bar(x=df_top["Ville"], y=df_top["Nombre de demandes"], marker_color='indianred')])
                fig.update_layout(title="Nombre de demandes par ville", xaxis_title="Ville", yaxis_title="Nombre de demandes", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Erreur lors de la récupération des données par ville.")
        except Exception as e:
            st.error("Impossible de joindre l'endpoint /top_cities.")
    with tab3:
        st.header("Feedback")
        with st.form("feedback_form"):
            rating = st.slider("Votre note (1 à 5)", min_value=1, max_value=5, value=3)
            comment = st.text_area("Votre commentaire (facultatif)")
            submitted = st.form_submit_button("Envoyer le feedback")
            if submitted:
                feedback_url = "http://localhost:8000/feedback"
                data = {"rating": rating, "comment": comment}
                response = requests.post(feedback_url, data=data)
                if response.status_code == 200:
                    st.success("Merci pour votre retour.")
                else:
                    st.error("Erreur lors de l'envoi du feedback.")

if __name__ == "__main__":
    run_streamlit()
