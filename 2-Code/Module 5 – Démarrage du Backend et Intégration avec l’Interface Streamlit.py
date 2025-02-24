def run_backend():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Démarrage du backend en arrière-plan si ce n'est déjà fait.
if "backend_started" not in st.session_state:
    st.session_state.backend_started = True
    threading.Thread(target=run_backend, daemon=True).start()
    time.sleep(1)

st.title("Application Météo – Commande vocale / manuelle (Open-Meteo)")

if "forecast_response" not in st.session_state:
    st.session_state["forecast_response"] = None

# Construction de l'interface en trois onglets, permettant de gérer les prévisions, l'analyse et les feedbacks.
tab1, tab2, tab3 = st.tabs(["Prévisions", "Analyse & Monitoring", "Feedback"])

with tab1:
    st.header("Bienvenu !")
    mode = st.radio("Choisissez votre mode de commande :", ("Enregistrement par micro", "Charger un fichier audio", "Manuelle"))
    
    audio_bytes = None
    transcription_input = ""
    city_input = ""
    forecast_days_input = None
    if mode != "Enregistrement par micro":
        forecast_days_input = st.selectbox("Nombre de jours de prévision", options=[3, 5, 7], index=2)
    
    if mode == "Enregistrement par micro":
        st.subheader("Enregistrement par micro")
        if st.button("Démarrer la transcription vocale"):
            transcription_input = azure_speech_from_microphone()
            st.write("Transcription obtenue :", transcription_input)
            backend_url = "http://localhost:8000/process_command"
            data = {"transcription": transcription_input, "mode": "vocale"}
            if city_input:
                data["city"] = city_input
            response = requests.post(backend_url, data=data)
            if response.status_code == 200:
                result = response.json()
                st.session_state.forecast_response = result
                st.success(f"Prévision pour {result['location']} (commande vocale)")
            else:
                st.error(f"Erreur (code {response.status_code}) lors de la commande vocale")
        city_input = st.text_input("Ville (optionnel)")
    elif mode == "Charger un fichier audio":
        st.subheader("Charger un fichier audio")
        uploaded_file = st.file_uploader("Sélectionnez votre fichier audio", type=["wav", "mp3"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            st.audio(audio_bytes, format="audio/wav")
        transcription_input = st.text_input("Transcription (optionnelle)")
        city_input = st.text_input("Ville (optionnel)")
    else:
        st.subheader("Commande manuelle")
        city_input = st.text_input("Ville")
    
    if mode != "Enregistrement par micro" and st.button("Envoyer la commande"):
        backend_url = "http://localhost:8000/process_command"
        files = {}
        data = {}
        if audio_bytes is not None:
            files["file"] = ("audio.wav", audio_bytes, "audio/wav")
        if transcription_input:
            data["transcription"] = transcription_input
        if city_input:
            data["city"] = city_input
        if forecast_days_input is not None:
            data["forecast_days"] = str(forecast_days_input)
        data["mode"] = "manuel"
        try:
            response = requests.post(backend_url, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prévision pour {result['location']}")
                st.session_state.forecast_response = result
            else:
                st.error(f"Erreur (code {response.status_code}) lors de l'envoi de la commande")
        except Exception as e:
            st.error("Impossible de joindre le backend. Vérifiez qu'il est démarré et accessible.")
    
    if st.session_state.forecast_response and st.button("Afficher les résultats de la commande vocale"):
        result = st.session_state.forecast_response
        st.subheader("Prévisions actuelles")
        final_days = result["forecast_days"]
        df = pd.DataFrame(result["forecast"]["hourly"])
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df_filtered = df[df['hour'] == 12].sort_values(by='date').head(final_days)
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Température (°C)", "Ciel (%)", "Vent (km/h)"))
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['temperature_2m'],
                                 mode='lines+markers', marker=dict(color='red')),
                                 row=1, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['cloudcover'],
                                 mode='lines+markers', marker=dict(color='blue')),
                                 row=2, col=1)
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['windspeed_10m'],
                                 mode='lines+markers', marker=dict(color='green')),
                                 row=3, col=1)
        fig.update_layout(height=600, title_text="Prévisions de Midi sur {} jours".format(final_days), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Détails des prévisions")
        st.dataframe(df_filtered[['date', 'temperature_2m', 'cloudcover', 'windspeed_10m', 'pm2_5']].rename(
            columns={
                "date": "Date",
                "temperature_2m": "Température (°C)",
                "cloudcover": "Ciel (%)",
                "windspeed_10m": "Vent (km/h)",
                "pm2_5": "Pollution (µg/m³)"
            }
        ))
        st.subheader("Localisation de la ville")
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
            st.write("Retours utilisateurs :", len(data_analysis.get("feedbacks", [])))
        else:
            st.error("Erreur lors de la récupération des données d'analyse.")
    except Exception as e:
        st.error("Impossible de joindre le backend pour l'analyse.")
    
    try:
        metrics_url = "http://localhost:8000/metrics"
        metrics_response = requests.get(metrics_url)
        if metrics_response.status_code == 200:
            st.subheader("Métriques Prometheus")
            lines = metrics_response.text.splitlines()
            definitions = {
                "http_requests_total": "Nombre total de requêtes HTTP, ventilées par méthode, endpoint et code HTTP.",
                "http_request_duration_seconds": "Durée des requêtes HTTP par méthode et endpoint.",
                "forecast_requests_total": "Nombre total de demandes de prévisions traitées.",
                "errors_total": "Nombre total d'erreurs survenues.",
                "feedback_total": "Nombre total de retours utilisateurs enregistrés."
            }
            metrics_data = []
            for key, definition in definitions.items():
                for line in lines:
                    if line.startswith(key):
                        parts = line.split()
                        if len(parts) >= 2:
                            metric_value = parts[1]
                            metrics_data.append({"Metric": key, "Value": metric_value})
                        break
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                df_metrics_styled = df_metrics.style.set_properties(subset=["Value"], **{'min-width': '300px'})
                st.dataframe(df_metrics_styled)
                st.markdown("**Définitions des métriques :**")
                for key, definition in definitions.items():
                    st.markdown(f"- **{key}** : {definition}")
            else:
                st.write("Aucune métrique trouvée.")
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
            df_top = df_top.sort_values(by="Nombre de demandes", ascending=False)
            st.dataframe(df_top.style.set_properties(**{'text-align': 'center', 'min-width': '200px'}))
            fig = go.Figure(data=[go.Bar(x=df_top["Ville"], y=df_top["Nombre de demandes"], marker_color='indianred')])
            fig.update_layout(title="Nombre de demandes par ville",
                              xaxis_title="Ville",
                              yaxis_title="Nombre de demandes",
                              template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Erreur lors de la récupération des données par ville.")
    except Exception as e:
        st.error("Impossible de joindre l'endpoint /top_cities.")

with tab3:
    st.header("Feedback")
    with st.form("feedback_form"):
        rating = st.slider("Votre note (1-5)", min_value=1, max_value=5, value=3)
        comment = st.text_area("Votre commentaire (facultatif)")
        submitted = st.form_submit_button("Envoyer votre feedback")
        if submitted:
            feedback_url = "http://localhost:8000/feedback"
            data = {"rating": rating, "comment": comment}
            response = requests.post(feedback_url, data=data)
            if response.status_code == 200:
                st.success("Merci pour votre feedback !")
            else:
                st.error("Erreur lors de l'envoi du feedback.")
