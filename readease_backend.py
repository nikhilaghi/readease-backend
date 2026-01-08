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
    mode: str = Field(default="summary")  

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
        "purpose": "AI-powered accessible reading comprehension API for students",
        "gtts_ready": GTTS_AVAILABLE,
        "azure_ready": AZURE_AVAILABLE,
        "endpoints": [
            "/health",
            "/simplify",
            "/key-ideas",
            "/process",
            "/speak"
        ]
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

def trim_bad_ending(sentence: str) -> str:
    bad_endings = {
        "and", "but", "or", "which", "that", "because", "while", "so"
    }

    words = sentence.split()
    while words and words[-1].lower().strip(",.;:") in bad_endings:
        words.pop()

    return " ".join(words)

def close_unmatched_parenthesis(sentence: str) -> str:
    if sentence.count("(") > sentence.count(")"):
        sentence = sentence.rsplit("(", 1)[0].strip()
    return sentence

ACADEMIC_SIMPLIFY_MAP = {
    "ontological": "about existence",
    "epistemological": "about knowledge",
    "ephemeral": "short-lived",
    "perennial": "long-lasting",
    "hermeneutics": "interpretation",
    "paradox": "contradiction",
    "recalibration": "adjustment",
    "discourse": "discussion",
    "verity": "truth"
}

VOCABULARY_DB = {
    "ontological": {
        "meaning": "related to the nature of existence",
        "example": "The ontological question asks what truly exists."
    },
    "epistemological": {
        "meaning": "related to the study of knowledge",
        "example": "Epistemological debates focus on how we know things."
    },
    "ephemeral": {
        "meaning": "lasting for a very short time",
        "example": "Online trends are often ephemeral."
    },
    "perennial": {
        "meaning": "lasting for a long time",
        "example": "Education is a perennial human need."
    },
    "hermeneutics": {
        "meaning": "the theory of interpretation",
        "example": "Hermeneutics helps interpret complex texts."
    },
    "paradox": {
        "meaning": "a statement that seems contradictory but may be true",
        "example": "It is a paradox that less is sometimes more."
    },
    "recalibration": {
        "meaning": "the process of adjusting something",
        "example": "The system needs recalibration after updates."
    },
    "labyrinthine": {
        "meaning": "very complex or confusing",
        "example": "The rules were labyrinthine and hard to follow."
    },
    "cognition": {
        "meaning": "the mental process of thinking and understanding",
        "example": "Reading improves cognition."
    }
}


def safe_sentence(sentence: str) -> str:
    bad_endings = {
        "and", "but", "or", "which", "that", "because",
        "while", "to", "has", "have", "had", "with",
        "also", "as", "by", "from"
    }

    words = sentence.split()
    while words and words[-1].lower().strip(",.;:") in bad_endings:
        words.pop()

    return " ".join(words)


def explain_text(text: str, level: str = "medium", mode: str = "summary"):
    # ---------- VALIDATION ----------
    if level not in {"simple", "medium", "hard"}:
        level = "medium"

    if mode not in {"summary", "rewrite"}:
        mode = "summary"

    if not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    # ---------- NORMALIZE ----------
    text = re.sub(r"\s+", " ", text).strip()

    # ---------- SENTENCE SPLIT ----------
    sentences = re.split(r'(?<=[.!?])\s+', text)

    simplified_sentences = []

    for sentence in sentences:
        # ---------- VOCAB SIMPLIFICATION ----------
        sentence = simplify_vocabulary(sentence)
        for word, simple in ACADEMIC_SIMPLIFY_MAP.items():
            sentence = re.sub(
                rf"\b{word}\b",
                simple,
                sentence,
                flags=re.IGNORECASE
            )

        # ---------- BREAK LONG SENTENCES ----------
        if len(sentence.split()) > 28:
            parts = re.split(
                r",|;|—|\bwhich\b|\bthat\b|\bbecause\b|\bwhile\b",
                sentence,
                flags=re.IGNORECASE
            )
        else:
            parts = [sentence]

        for part in parts:
            part = part.strip()
            part = safe_sentence(part)

            if len(part.split()) < 6:
                continue

            part = part.capitalize()
            if not part.endswith(('.', '!', '?')):
                part += "."

            simplified_sentences.append(part)

    # ---------- SUMMARY MODE ----------
    if mode == "summary":
        if level == "simple":
            final_text = "\n".join(f"• {s}" for s in simplified_sentences[:5])
        else:
            final_text = " ".join(simplified_sentences[:6])

    # ---------- REWRITE MODE ----------
    else:
        if level == "simple":
            final_text = "\n".join(f"• {s}" for s in simplified_sentences)
        else:
            final_text = " ".join(simplified_sentences)

    # ---------- FALLBACK ----------
    if not final_text.strip():
        final_text = "The text could not be simplified meaningfully."

    return {
        "text": final_text,
        "reading_time_minutes": estimate_reading_time(final_text),
        "estimated_grade_level": estimate_grade_level(final_text)
    }




def extract_key_ideas(text: str) -> list:
    text = text.lower()

    verb_blacklist = {
        "is", "are", "was", "were", "be", "being", "been",
        "have", "has", "had", "do", "does", "did",
        "make", "makes", "made", "use", "using",
        "require", "requires", "needed", "need"
    }

    stop_words = {
        "the", "and", "of", "to", "for", "with", "in",
        "on", "at", "by", "from", "this", "that"
    }

    words = re.findall(r"\b[a-z]{3,}\b", text)

    phrases = []

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # skip junk
        if w1 in stop_words or w2 in stop_words:
            continue
        if w1 in verb_blacklist or w2 in verb_blacklist:
            continue

        phrase = f"{w1} {w2}"
        phrases.append(phrase)

    if not phrases:
        return ["Key concepts not detected"]

    from collections import Counter
    most_common = Counter(phrases).most_common(5)

    return [phrase.title() for phrase, _ in most_common]

def extract_vocabulary(text: str, limit: int = 5):
    words = re.findall(r"\b[a-zA-Z]{7,}\b", text.lower())

    seen = set()
    vocab_list = []

    for word in words:
        if word in seen:
            continue
        seen.add(word)

        if word in VOCABULARY_DB:
            entry = VOCABULARY_DB[word]
            vocab_list.append({
                "word": word,
                "meaning": entry["meaning"],
                "example": entry["example"]
            })

        if len(vocab_list) == limit:
            break

    return vocab_list

@app.post("/simplify")
async def simplify_text(data: TextInput):
    return {
        "simplified_text": explain_text(data.text, data.level),
        "status": "success"
    }

@app.post("/key-ideas")
async def key_ideas(data: TextInput):
    return {"key_ideas": extract_key_ideas(data.text), "status": "success"}

@app.post("/vocabulary")
async def vocabulary_trainer(data: TextInput):
    return {
        "daily_words": extract_vocabulary(data.text),
        "status": "success"
    }

@app.post("/process")
async def process_text(data: TextInput):
    explanation = explain_text(data.text, data.level, data.mode)

    return {
        "simplified_text": explanation["text"],
        "reading_time_minutes": explanation["reading_time_minutes"],
        "estimated_grade_level": explanation["estimated_grade_level"],
        "key_ideas": extract_key_ideas(data.text),
        "vocabulary": extract_vocabulary(data.text),
        "accessibility": {
            "tts_available": GTTS_AVAILABLE or AZURE_AVAILABLE
        },
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
        except Exception:
            pass

    if AZURE_AVAILABLE:
        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=SPEECH_KEY,
                region=SPEECH_REGION
            )
            speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
            synthesizer = speechsdk.SpeechSynthesizer(speech_config)

            result = synthesizer.speak_text_async(data.text).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return StreamingResponse(
                    io.BytesIO(result.audio_data),
                    media_type="audio/wav"
                )
        except Exception:
            pass

    raise HTTPException(
        status_code=503,
        detail="Text-to-speech service temporarily unavailable"
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "readease_backend:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )