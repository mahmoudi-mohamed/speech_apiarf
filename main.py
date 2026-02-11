from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from piper import PiperVoice
import wave
import base64
import tempfile
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
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
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker embeddings of a lovely male voice
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

class TextRequest(BaseModel):
    text: str

@app.post("/tts")
def text_to_speech(request: TextRequest):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav_file:
        wav_file_path = tmp_wav_file.name
        inputs = processor(text=request.text, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        sf.write(wav_file_path, speech.numpy(), samplerate=16000)

        with open(wav_file_path, "rb") as f:
            audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return {"audio_base64": audio_b64}

@app.get("/")
def root():
    return {"message": "Arabic TTS API is running!"}
