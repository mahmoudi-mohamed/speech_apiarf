from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import base64

app = FastAPI(title="Arabic TTS API")

# تحميل الموديل
tts = pipeline("text-to-speech", "Reyouf/speecht5_tts_Arabic")

class TextRequest(BaseModel):
    text: str

@app.post("/tts")
def text_to_speech(request: TextRequest):
    output = tts(request.text)
    audio_bytes = output["audio"]
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio_base64": audio_b64}

@app.get("/")
def root():
    return {"message": "Arabic TTS API is running!"}
