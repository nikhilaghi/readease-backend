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


@app.api_route("/health", methods=["GET", "HEAD"])
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
        "meaning": "related to the nature of existence or being",
        "example": "The ontological question asks what truly exists."
    },
    "epistemological": {
        "meaning": "related to the study of knowledge and how we know things",
        "example": "Epistemological debates focus on the sources of knowledge."
    },
    "discourse": {
        "meaning": "formal discussion or written communication",
        "example": "Academic discourse often uses specialized language."
    },
    "paradox": {
        "meaning": "a situation or statement that seems contradictory but may be true",
        "example": "It is a paradox that less can sometimes be more."
    },
    "cognition": {
        "meaning": "the mental process of thinking, understanding, and learning",
        "example": "Reading regularly improves cognition."
    },
    "framework": {
        "meaning": "a basic structure used to support ideas or theories",
        "example": "The framework helps organize complex concepts."
    },
    "theoretical": {
        "meaning": "based on ideas rather than practical experience",
        "example": "The paper presents a theoretical model of learning."
    },
    "comprehension": {
        "meaning": "the ability to understand something",
        "example": "Good comprehension is essential for academic success."
    },
    "interdisciplinary": {
        "meaning": "involving two or more academic disciplines",
        "example": "The course takes an interdisciplinary approach."
    },
    "methodology": {
        "meaning": "a system of methods used in research or study",
        "example": "The research methodology is clearly explained."
    },
    "cognitive": {
        "meaning": "related to mental processes like thinking and memory",
        "example": "Cognitive skills affect how students learn."
    },
    "articulate": {
        "meaning": "to express ideas clearly and effectively",
        "example": "She articulated her argument well."
    },
    "implicit": {
        "meaning": "suggested without being directly stated",
        "example": "The rules were implicit rather than written."
    },
    "explicit": {
        "meaning": "clearly stated and easy to understand",
        "example": "The instructions were explicit."
    },
    "interpretation": {
        "meaning": "the act of explaining the meaning of something",
        "example": "Different readers may have different interpretations."
    },
    "complexity": {
        "meaning": "the state of having many parts or being difficult to understand",
        "example": "The complexity of the text made it hard to read."
    },
    "precision": {
        "meaning": "accuracy and exactness",
        "example": "Academic writing values precision."
    },
    "abstraction": {
        "meaning": "an idea that is not concrete or physical",
        "example": "Abstraction is common in theoretical subjects."
    },
    "synthesis": {
        "meaning": "combining different ideas to form a whole",
        "example": "The essay demonstrates synthesis of multiple sources."
    }
}

VOCABULARY_DB.update({

    "analysis": {
        "meaning": "a detailed examination to understand something better",
        "example": "The analysis of the data revealed important trends."
    },

    "evaluation": {
        "meaning": "judging the quality or value of something",
        "example": "The evaluation measured student performance."
    },

    "hypothesis": {
        "meaning": "a testable explanation or prediction",
        "example": "The hypothesis was tested through experiments."
    },

    "inference": {
        "meaning": "a conclusion drawn from evidence",
        "example": "An inference was made based on the results."
    },

    "correlation": {
        "meaning": "a relationship between two variables",
        "example": "There is a correlation between sleep and learning."
    },

    "significance": {
        "meaning": "importance or meaning of something",
        "example": "The findings have scientific significance."
    },

    "assumption": {
        "meaning": "something believed to be true without proof",
        "example": "The model relies on several assumptions."
    },

    "limitation": {
        "meaning": "a weakness or restriction in a study",
        "example": "A limitation of the study was the small sample size."
    },

    "perspective": {
        "meaning": "a way of viewing or thinking about something",
        "example": "The issue was analyzed from a social perspective."
    },

    "empirical": {
        "meaning": "based on observation or experiment",
        "example": "The theory is supported by empirical evidence."
    },

    "methodological": {
        "meaning": "related to research methods",
        "example": "The paper discusses methodological challenges."
    },

    "consistency": {
        "meaning": "the quality of being reliable and stable",
        "example": "The experiment showed consistency across trials."
    },

    "relevance": {
        "meaning": "how closely something is related to the topic",
        "example": "The topic has strong relevance to education."
    },

    "validation": {
        "meaning": "confirmation that something is correct",
        "example": "Further validation of the model is required."
    },

    "implementation": {
        "meaning": "putting a plan or idea into action",
        "example": "Implementation of the policy begins next year."
    },

    "optimization": {
        "meaning": "making something as effective as possible",
        "example": "The algorithm focuses on optimization."
    },

    "constraint": {
        "meaning": "a limitation or restriction",
        "example": "Time was a major constraint in the project."
    },

    "generalization": {
        "meaning": "a broad conclusion drawn from specific cases",
        "example": "Generalization should be done carefully in research."
    }

})


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


def is_fragment(sentence: str) -> bool:
    bad_starts = (
        "and ", "but ", "or ", "especially ", "which ",
        "that ", "because ", "while ", "not only ",
        "also ", "particularly ", "such as "
    )
    return sentence.lower().startswith(bad_starts)


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

       # ---------- SPLIT LONG SENTENCES ----------
            if mode == "summary" and len(sentence.split()) > 28:
                parts = re.split(
                    r",|;|—|\bwhich\b|\bthat\b|\bbecause\b|\bwhile\b",
                    sentence,
                    flags=re.IGNORECASE
                    )

        else:
            parts = [sentence]

        for part in parts:
            part = part.strip()

            # 🚫 Reject fragments
            if is_fragment(part):
                continue

            part = safe_sentence(part)

            # 🚫 Reject too short / weak clauses
            if mode == "summary" and len(part.split()) < 8:
                 continue


            part = part.capitalize()
            if not part.endswith(('.', '!', '?')):
                part += "."

            simplified_sentences.append(part)

    # ---------- SUMMARY / REWRITE ----------
    if mode == "summary":
        if level == "simple":
            final_text = "\n".join(f"• {s}" for s in simplified_sentences[:5])
        else:
            final_text = " ".join(simplified_sentences[:6])
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

    if not vocab_list:
        return [{
            "word": "—",
            "meaning": "No advanced academic terms detected in this text.",
            "example": "Try pasting a more technical paragraph."
        }]

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