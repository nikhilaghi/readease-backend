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
    level: str = Field(default="medium")

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

def estimate_reading_time(text: str) -> int:
    words = len(text.split())
    return max(1, round(words / 200)) 


def estimate_grade_level(text: str) -> str:
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))

    if avg_word_len < 4.5:
        return "Grade 4–6"
    elif avg_word_len < 5.5:
        return "Grade 7–9"
    elif avg_word_len < 6.5:
        return "Grade 10–12"
    else:
        return "College level"



def explain_text(text: str, level: str = "medium"):
    if level not in {"simple", "medium", "hard"}:
        level = "medium"

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    sentences = re.split(r'(?<=[.!?])\s+', text)
    output = []

    for sentence in sentences:
        sentence = simplify_vocabulary(sentence)
        words = sentence.split()

        if level == "simple":
            if len(words) < 8:
                continue

            sentence = sentence.replace(",", "").replace(";", "")
            sentence = re.sub(r'^(and|but|so)\s+', '', sentence, flags=re.IGNORECASE)
            sentence = ' '.join(words[:14]).rstrip(",;:") + "."
            output.append(f"• {sentence.capitalize()}")

        elif level == "medium":
            if len(words) < 10:
                continue

            if len(words) > 30:
                sentence = ' '.join(words[:25]).rstrip(",;:") + "."
            else:
                sentence = sentence.rstrip(",;:")

            sentence = sentence.strip()
            sentence = sentence[0].upper() + sentence[1:]

            if not sentence.endswith(('.', '!', '?')):
                sentence += "."

            output.append(sentence)

        else:
            if len(words) < 12:
                continue

            sentence = sentence.strip()
            sentence = sentence[0].upper() + sentence[1:]

            if not sentence.endswith(('.', '!', '?')):
                sentence += "."

            output.append(sentence)

        if len(output) == (5 if level == "simple" else 3):
            break

    final_text = "\n".join(output) if level == "simple" else " ".join(output)

    return {
        "text": final_text,
        "reading_time_minutes": estimate_reading_time(final_text),
        "estimated_grade_level": estimate_grade_level(final_text)
    }





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
    return {
        "simplified_text": explain_text(data.text, data.level),
        "status": "success"
    }


@app.post("/key-ideas")
async def key_ideas(data: TextInput):
    return {"key_ideas": extract_key_ideas(data.text), "status": "success"}

@app.post("/process")
async def process_text(data: TextInput):
    explanation = explain_text(data.text, data.level)

    return {
        "simplified_text": explanation["text"],
        "reading_time_minutes": explanation["reading_time_minutes"],
        "estimated_grade_level": explanation["estimated_grade_level"],
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
  

