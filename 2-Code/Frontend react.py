import React, { useState } from 'react';
import axios from 'axios';
import { ReactMic } from 'react-mic';
import './App.css'; // Vous pouvez ajouter votre style ici

// Composant pour la commande vocale
function VoiceCommand() {
  const [record, setRecord] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [result, setResult] = useState(null);

  const startRecording = () => {
    setRecord(true);
    setResult(null);
  };

  const stopRecording = () => {
    setRecord(false);
  };

  const onStop = (recordedData) => {
    setRecordedBlob(recordedData);
  };

  const sendRecording = async () => {
    if (!recordedBlob) {
      alert("Aucun enregistrement n'est disponible !");
      return;
    }
    const formData = new FormData();
    // recordedBlob.blob contient l'objet Blob du fichier audio
    formData.append('file', recordedBlob.blob, 'recording.wav');

    try {
      const response = await axios.post('http://localhost:8000/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'api_key': 'secret-api-key'
        }
      });
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert('Erreur lors de l\'envoi de l\'enregistrement.');
    }
  };

  return (
    <div>
      <h2>Commande Vocale</h2>
      <ReactMic
        record={record}
        className="sound-wave"
        onStop={onStop}
        strokeColor="#000000"
        backgroundColor="#FF4081"
      />
      <div style={{ marginTop: '1rem' }}>
        <button onClick={startRecording}>Démarrer l'enregistrement</button>
        <button onClick={stopRecording}>Arrêter l'enregistrement</button>
        <button onClick={sendRecording}>Envoyer la commande vocale</button>
      </div>
      {result && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Résultat</h3>
          <p><strong>Lieu :</strong> {result.location}</p>
          <p><strong>Horizon temporel :</strong> {result.time_horizon}</p>
          <pre>{JSON.stringify(result.forecast, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

// Composant pour l'analyse et le monitoring
function AnalysisMonitoring() {
  const [queryId, setQueryId] = useState('');
  const [actualTemp, setActualTemp] = useState('');
  const [feedbackResponse, setFeedbackResponse] = useState(null);

  const sendFeedback = async () => {
    const payload = {
      query_id: queryId,
      actual_forecast: { main: { temp: parseFloat(actualTemp) } }
    };
    try {
      const response = await axios.post('http://localhost:8000/feedback', payload, {
        headers: {
          'Content-Type': 'application/json',
          'api_key': 'secret-api-key'
        }
      });
      setFeedbackResponse(response.data);
    } catch (error) {
      console.error(error);
      alert('Erreur lors de l\'envoi du feedback.');
    }
  };

  return (
    <div>
      <h2>Analyse et Monitoring</h2>
      <div style={{ marginBottom: '1rem' }}>
        <h3>Feedback et Comparaison des Prévisions</h3>
        <label>
          Identifiant de la requête (query_id) :
          <input
            type="text"
            value={queryId}
            onChange={(e) => setQueryId(e.target.value)}
            placeholder="Identifiant fourni lors de la commande"
          />
        </label>
      </div>
      <div style={{ marginBottom: '1rem' }}>
        <label>
          Température réelle observée (°C) :
          <input
            type="number"
            value={actualTemp}
            onChange={(e) => setActualTemp(e.target.value)}
          />
        </label>
      </div>
      <button onClick={sendFeedback}>Envoyer mon Feedback</button>
      {feedbackResponse && (
        <div style={{ marginTop: '1rem' }}>
          <h3>Réponse du Feedback</h3>
          <pre>{JSON.stringify(feedbackResponse, null, 2)}</pre>
        </div>
      )}
      <div style={{ marginTop: '2rem' }}>
        <h3>Tableau de Bord Analytique</h3>
        <p>Section de visualisation des performances (à compléter selon vos données réelles).</p>
      </div>
    </div>
  );
}

// Composant principal avec onglets
function App() {
  const [activeTab, setActiveTab] = useState('commande');

  return (
    <div className="App">
      <header>
        <h1>Prévisions Météo par Commande Vocale et Analyse de Performance</h1>
        <nav style={{ margin: '1rem 0' }}>
          <button onClick={() => setActiveTab('commande')}>
            Commande Vocale
          </button>
          <button onClick={() => setActiveTab('analyse')}>
            Analyse et Monitoring
          </button>
        </nav>
      </header>
      <main>
        {activeTab === 'commande' && <VoiceCommand />}
        {activeTab === 'analyse' && <AnalysisMonitoring />}
      </main>
    </div>
  );
}

export default App;
