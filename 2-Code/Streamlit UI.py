import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Météo & Analyse", layout="wide")
st.title("Prévisions Météo par Commande Vocale et Analyse de Performance")

# Création de deux onglets : l'un pour la commande vocale et l'autre pour l'analyse/monitoring
tab_commande, tab_analyse = st.tabs(["Commande Vocale", "Analyse et Monitoring"])

with tab_commande:
    st.header("Commande Vocale")
    st.write("Utilisez le module ci-dessous pour enregistrer votre commande vocale et obtenir la prévision correspondante.")
    
    uploaded_file = st.file_uploader("Sélectionnez votre fichier audio (formats acceptés : wav, mp3, m4a)", type=["wav", "mp3", "m4a"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Envoyer ma commande vocale"):
            files = {"file": uploaded_file}
            headers = {"api_key": "secret-api-key"}  # Veuillez remplacer par votre clé d'accès réelle
            try:
                response = requests.post("http://localhost:8000/process", files=files, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    st.success("Prévision reçue avec succès.")
                    st.markdown(f"**Lieu :** {data['location']}  \n**Horizon temporel :** {data['time_horizon']}")
                    st.write("### Détails de la prévision")
                    st.json(data["forecast"])
                    if "main" in data["forecast"]:
                        temp = data["forecast"]["main"].get("temp")
                        st.write(f"Température actuelle : {temp} °C")
                    # Affichage d'une carte interactive (démonstration : Paris ou coordonnées par défaut)
                    if data["location"].lower() == "paris":
                        lat, lon = 48.8566, 2.3522
                    else:
                        lat, lon = 0, 0
                    df_map = pd.DataFrame({"lat": [lat], "lon": [lon]})
                    st.map(df_map)
                else:
                    st.error("Erreur lors de l'appel à l'API : " + response.text)
            except Exception as e:
                st.error("Une exception est survenue : " + str(e))

with tab_analyse:
    st.header("Analyse et Monitoring")
    st.write("Dans cet espace, vous pouvez contribuer à l’amélioration continue des prévisions en fournissant un feedback sur les valeurs réelles, et visualiser les performances du système à travers divers tableaux de bord analytiques.")
    
    st.subheader("Feedback et Comparaison des Prévisions")
    st.write("Pour affiner la précision des prédictions, indiquez ici la valeur réelle observée. Ce retour sera utilisé pour comparer avec la prévision initiale.")
    
    query_id = st.text_input("Identifiant de la requête (query_id)", placeholder="Saisissez l’identifiant fourni lors de la commande")
    actual_temp = st.number_input("Température réelle observée (en °C)", value=0.0, step=0.1)
    
    if st.button("Envoyer mon Feedback"):
        feedback_payload = {
            "query_id": query_id,
            "actual_forecast": {"main": {"temp": actual_temp}}
        }
        headers = {"api_key": "secret-api-key", "Content-Type": "application/json"}
        try:
            response = requests.post("http://localhost:8000/feedback", data=json.dumps(feedback_payload), headers=headers)
            if response.status_code == 200:
                st.success("Feedback envoyé avec succès.")
                st.json(response.json())
            else:
                st.error("Erreur lors de l'envoi du feedback : " + response.text)
        except Exception as e:
            st.error("Une exception est survenue lors de l'envoi du feedback : " + str(e))
    
    st.subheader("Tableau de Bord Analytique")
    st.write("Les graphiques ci-dessous présentent une simulation de l'évolution des prévisions par rapport aux valeurs réelles, permettant d'identifier d'éventuelles disparités et d'ajuster ultérieurement les modèles de prédiction.")
    
    # Simulation de données analytiques pour démonstration
    # Ces données pourraient être récupérées dynamiquement depuis un endpoint dédié en production
    dates = pd.date_range(start="2025-01-01", periods=10, freq='D')
    data = {
        "Date": dates,
        "Température Prévue": np.random.normal(loc=20, scale=2, size=10),
        "Température Réelle": np.random.normal(loc=20, scale=3, size=10)
    }
    df = pd.DataFrame(data)
    df["Erreur Absolue"] = abs(df["Température Prévue"] - df["Température Réelle"])
    
    st.write("### Comparaison des Températures Prévue vs Réelle")
    fig_line = px.line(df, x="Date", y=["Température Prévue", "Température Réelle"], markers=True,
                       title="Évolution Temporelle des Températures")
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.write("### Erreur de Prédiction")
    fig_bar = px.bar(df, x="Date", y="Erreur Absolue", title="Erreur Absolue par Date")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.write("### Statistiques de Performance")
    moyenne_erreur = df["Erreur Absolue"].mean()
    ecart_type_erreur = df["Erreur Absolue"].std()
    st.markdown(f"**Moyenne de l'erreur absolue :** {moyenne_erreur:.2f} °C")
    st.markdown(f"**Écart-type de l'erreur :** {ecart_type_erreur:.2f} °C")
