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
voice = PiperVoice.load("my_voice.onnx")

class TextRequest(BaseModel):
    text: str

@app.post("/tts")
def text_to_speech(request: TextRequest):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav_file:
        wav_file_path = tmp_wav_file.name
        voice.synthesize_wav(request.text, wav_file_path)

        with open(wav_file_path, "rb") as f:
            audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio_base64": audio_b64}

@app.get("/")
def root():
    return {"message": "Arabic TTS API is running!"}
