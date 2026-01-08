from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, JSONResponse
import re
import io
import uvicorn
import os

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("pip3 install gtts for reliable speech")

SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("pip3 install azure-cognitiveservices-speech")

app = FastAPI(title="ReadEase API - Fixed TTS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str = Field(..., max_length=2000, min_length=1)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"status": "error", "message": "Invalid input data"})

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error"})

@app.get("/")
async def home():
    return {
        "status": "running ✅",
        "gtts_ready": GTTS_AVAILABLE,
        "azure_ready": AZURE_AVAILABLE,
        "endpoints": ["/health", "/simplify", "/key-ideas", "/process", "/speak"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

def simplify_vocabulary(text: str) -> str:
    replacements = {
        r"\b(neural networks)\b": "AI systems",
        r"\b(quantum entanglement)\b": "linked particles",
        r"\b(paradigm shift)\b": "big change",
        r"\b(unprecedented)\b": "never before",
        r"\b(fundamentally)\b": "basically",
        r"\b(sophisticated)\b": "advanced",
        r"\b(computational)\b": "computer",
        r"\b(algorithm)\b": "method",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def clean_output(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text and not re.search(r"[.!?]$", text):
        text += "."
    return text

    
def explain_text(text: str) -> str:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    sentences = re.split(r'(?<=[.!?])\s+', text)
    simplified = []

    for sentence in sentences:
        if len(sentence.split()) < 12:
            continue

        sentence = simplify_vocabulary(sentence)

        clauses = re.split(r'[;,]', sentence)
        selected = []

        for clause in clauses:
            clause = clause.strip()
            if len(clause.split()) >= 8:
                selected.append(clause)
            if len(selected) == 2:
                break

        if not selected:
            continue

        combined = '. '.join(selected).strip()

        if not combined.endswith(('.', '!', '?')):
            combined += '.'

        simplified.append(combined)

        if len(simplified) == 3:
            break

    final = ' '.join(simplified)
    final = re.sub(r'\s+\.', '.', final)
    return final




def extract_key_ideas(text: str) -> list:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = {'the', 'and', 'for', 'are', 'have', 'this', 'with', 'from'}
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    
    if not filtered:
        return ["No key ideas found"]
    
    from collections import Counter
    counter = Counter(filtered).most_common(5)
    return [f"{word.capitalize()}" for word, count in counter]

@app.post("/simplify")
async def simplify_text(data: TextInput):
    return {"simplified_text": explain_text(data.text), "status": "success"}

@app.post("/key-ideas")
async def key_ideas(data: TextInput):
    return {"key_ideas": extract_key_ideas(data.text), "status": "success"}

@app.post("/process")
async def process_text(data: TextInput):
    return {
        "simplified_text": explain_text(data.text),
        "key_ideas": extract_key_ideas(data.text),
        "status": "success"
    }

@app.post("/speak")
async def speak_text(data: TextInput):
    if len(data.text) > 2000:
        raise HTTPException(status_code=413, detail="Max 2000 chars")
    
    if GTTS_AVAILABLE:
        try:
            tts = gTTS(text=data.text, lang='en', slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return StreamingResponse(mp3_fp, media_type="audio/mpeg")
        except Exception as e:
            print(f"gTTS failed: {e}")
    
    if AZURE_AVAILABLE:
        try:
            speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
            speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config)
            
            result = synthesizer.speak_text_async(data.text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted and result.audio_data:
                return StreamingResponse(io.BytesIO(result.audio_data), media_type="audio/wav")
            else:
                raise HTTPException(status_code=500, detail=f"Azure failed: {result.reason}")
        except Exception as e:
            print(f"Azure failed: {e}")
    
    raise HTTPException(status_code=503, detail="No TTS available. pip3 install gtts")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "readease_backend:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
  

