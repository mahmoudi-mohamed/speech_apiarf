from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import base64
import tempfile
import soundfile as sf
import onnxruntime
import json
from pygoruut.pygoruut import Pygoruut
import numpy as np

app = FastAPI(title="Arabic TTS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load Piper ONNX model and config
PIPER_ONNX_MODEL_PATH = "piper-onnx-zayd0-arabic-diacritized.onnx"
PIPER_ONNX_MODEL_CONFIG_PATH = "piper-onnx-zayd0-arabic-diacritized.onnx.json"

try:
    piper_session = onnxruntime.InferenceSession(PIPER_ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    with open(PIPER_ONNX_MODEL_CONFIG_PATH, "r") as f:
        piper_config = json.load(f)
    pygoruut = Pygoruut()
    print(f"Piper ONNX model loaded successfully from {PIPER_ONNX_MODEL_PATH}")
except Exception as e:
    print(f"Error loading Piper ONNX model or Pygoruut: {e}")
    piper_session = None
    piper_config = None
    pygoruut = None

class TextRequest(BaseModel):
    text: str

@app.post("/tts")
def text_to_speech(request: TextRequest):
    if not piper_session or not pygoruut or not piper_config:
        return {"error": "Piper ONNX model, Pygoruut, or config is not loaded."}

    text = request.text
    print(f"Received text for TTS: {text}")

    # Phonemize the text
    phonemes_response = pygoruut.phonemize(language="Arabic", sentence=text)
    
    # Convert phonemes to phoneme IDs
    phoneme_id_map = piper_config["phoneme_id_map"]
    phoneme_ids = []
    for word in phonemes_response.Words:
        for p in word.Phonetic: # Iterate over individual phonemes in the phonetic string
            if p in phoneme_id_map:
                phoneme_ids.extend(phoneme_id_map[p])

    # Add sentence silence
    phoneme_ids.extend(phoneme_id_map["."])


    # Run inference
    input_ids = np.array(phoneme_ids, dtype=np.int64).reshape((1, -1))
    input_lengths = np.array([input_ids.shape[1]], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32) # noise_scale, length_scale, noise_w

    audio = piper_session.run(
        None,
        {
            "input": input_ids,
            "input_lengths": input_lengths,
            "scales": scales,
        },
    )[0]
    
    samplerate = piper_config["audio"]["sample_rate"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav_file:
        wav_file_path = tmp_wav_file.name
        sf.write(wav_file_path, audio.squeeze(), samplerate=samplerate)

        with open(wav_file_path, "rb") as f:
            audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio_base64": audio_b64}

@app.get("/")
def root():
    return {"message": "Arabic TTS API is running!"}
