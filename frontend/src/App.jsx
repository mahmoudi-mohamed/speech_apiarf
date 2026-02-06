import { useState } from 'react';
import axios from 'axios';
import './App.css'; // Assuming App.css will be used for styling

function App() {
  const [text, setText] = useState('');
  const [audioSrc, setAudioSrc] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const API_URL = 'https://speech-apiarf.onrender.com/tts'; // Your deployed API URL

  const handleSynthesize = async () => {
    if (!text.trim()) {
      setError('Please enter some text to synthesize.');
      return;
    }

    setIsLoading(true);
    setError('');
    setAudioSrc('');

    try {
      const response = await axios.post(API_URL, { text });
      const audioBase64 = response.data.audio_base64;
      setAudioSrc(`data:audio/wav;base64,${audioBase64}`);
    } catch (err) {
      console.error('Error synthesizing speech:', err);
      setError('Failed to synthesize speech. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Arabic TTS Client</h1>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter Arabic text here..."
          rows="5"
          cols="50"
          disabled={isLoading}
        />
        <button onClick={handleSynthesize} disabled={isLoading}>
          {isLoading ? 'Synthesizing...' : 'Synthesize Speech'}
        </button>

        {error && <p className="error-message">{error}</p>}

        {audioSrc && (
          <div className="audio-player">
            <h2>Playback:</h2>
            <audio controls autoPlay src={audioSrc}>
              Your browser does not support the audio element.
            </audio>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
