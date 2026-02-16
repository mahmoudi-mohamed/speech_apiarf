from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from piper import PiperVoice
import wave
import base64
import tempfile

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
try:
    voice = PiperVoice.load("piper-onnx-zayd0-arabic-diacritized.onnx")
    print("Piper ONNX model loaded successfully using piper library.")
except Exception as e:
    print(f"Error loading Piper ONNX model with piper library: {e}")
    voice = None

class TextRequest(BaseModel):
    text: str

@app.post("/tts")
def text_to_speech(request: TextRequest):
    if not voice:
        return {"error": "Piper Voice model is not loaded."}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav_file:
        wav_file_path = tmp_wav_file.name
        with wave.open(wav_file_path, "wb") as wav_file:
            voice.synthesize_wav(request.text, wav_file)

        with open(wav_file_path, "rb") as f:
            audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio_base64": audio_b64}

@app.get("/")
def root():
    return {"message": "Arabic TTS API is running!"}
