from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import wave
import base64
import tempfile
from transformers import VitsModel, AutoTokenizer

import torch
import soundfile as sf

app = FastAPI(title="Arabic TTS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# تحميل الموديل
model = VitsModel.from_pretrained("wasmdashai/vits-ar")
tokenizer = AutoTokenizer.from_pretrained("wasmdashai/vits-ar")

class TextRequest(BaseModel):
    text: str

@app.post("/tts")
def text_to_speech(request: TextRequest):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav_file:
        wav_file_path = tmp_wav_file.name
        inputs = tokenizer(request.text, return_tensors="pt")
        with torch.no_grad():
            speech = model(**inputs).waveform
        sf.write(wav_file_path, speech.numpy(), samplerate=model.config.sampling_rate)

        with open(wav_file_path, "rb") as f:
            audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio_base64": audio_b64}

@app.get("/")
def root():
    return {"message": "Arabic TTS API is running!"}
