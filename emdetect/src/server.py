import os
import time
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ml.engine import PsychologyEngine
from ml.loader import MarkLoader


# Logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# API key auth
# Read key from env var EMDETECT_API_KEY.
# If not set, a random key is generated at startup and printed to stdout.

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _load_api_key() -> str:
    key = os.environ.get("EMDETECT_API_KEY", "").strip()
    if not key:
        key = secrets.token_urlsafe(32)
        logger.warning(f"EMDETECT_API_KEY not set. Using generated key: {key}")
        logger.warning("Set EMDETECT_API_KEY env var to use a fixed key.")
    return key


API_KEY = _load_api_key()


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# App state — engine and tags loaded once at startup
class AppState:
    engine:     PsychologyEngine | None = None
    tags_data:  list[dict]              = []
    ready:      bool                    = False
    started_at: float                   = 0.0

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading MiniLM L12...")
    state.engine     = PsychologyEngine(data_dir=DATA_DIR)
    loader           = MarkLoader(state.engine)
    state.tags_data  = loader.load_tags_from_config(CONFIG_PATH)
    state.ready      = bool(state.tags_data)
    state.started_at = time.time()
    logger.info(f"Ready. Categories loaded: {len(state.tags_data)}")
    yield
    logger.info("Shutting down.")

# FastAPI app

app = FastAPI(
    title="EmDetect",
    version="2.1.0",
    description="Multilingual semantic emotional state analyser.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)



SUPPORTED_LANGS = {"ru", "uk", "en", "de"}


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    lang: str | None = Field(
        default=None,
        description="Language code: ru / uk / en / de. Auto-detected if omitted."
    )


class CategoryResult(BaseModel):
    category: str
    score:    float
    level:    str   # "high" | "medium" | "low"


class AnalyzeResponse(BaseModel):
    lang:     str
    results:  list[CategoryResult]
    detected: bool  # False if no categories passed threshold


# Language detection


def detect_lang(text: str) -> str:
    try:
        from langdetect import detect
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "ru"
    except Exception:
        return "ru"


# Routes


@app.get("/health", tags=["system"])
def health():
    # Public — no auth required, useful for uptime monitors
    return {
        "status": "ok" if state.ready else "loading",
        "ready": state.ready,
        "categories": len(state.tags_data),
        "uptime_sec": round(time.time() - state.started_at, 1),
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(req: AnalyzeRequest, _: None = Security(require_api_key)):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Engine not ready yet.")

    if req.lang:
        lang = req.lang.lower()
        if lang not in SUPPORTED_LANGS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported language '{lang}'. Supported: {sorted(SUPPORTED_LANGS)}"
            )
    else:
        lang = detect_lang(req.text)

    matches = state.engine.get_top_matches(req.text, lang, state.tags_data)

    results = [
        CategoryResult(
            category=m["tag_id"],
            score=m["score"],
            level="high" if m["score"] > 0.85 else "medium" if m["score"] > 0.70 else "low"
        )
        for m in matches
    ]

    return AnalyzeResponse(lang=lang, results=results, detected=bool(results))
