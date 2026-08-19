"""
config.py — all environment reads and defaults for ai-live-orchestrator.
Extracted from server.py so configuration stays separate from the pipeline.
All os.environ.get calls are clean (no trailing spaces — that was an old
critical bug which silently ignored every .bat setting).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


# ── helpers ──────────────────────────────────────────────────────
def env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ── backend / model ─────────────────────────────────────────────
LLM_BACKEND = "llama_cpp"
MODEL_PATH = env_str("MODEL_PATH", str(PROJECT_ROOT / "models" / "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"))
MODEL_LABEL = env_str("MODEL_LABEL", Path(MODEL_PATH).name)
LAUNCHER_NAME = env_str("LAUNCHER_NAME", "unknown.bat")

LLAMA_HOST = env_str("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = env_int("LLAMA_PORT", 8080)
LLAMA_BASE_URL = env_str("LLAMA_BASE_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1").rstrip("/")
LLAMA_MODEL = env_str("LLAMA_MODEL", env_str("LLAMA_MODEL_ID", "local-gemma"))
LLAMA_API_KEY = env_str("LLAMA_API_KEY", "no-key")
LLAMA_AUTO_START = env_bool("LLAMA_AUTO_START", False)
LLAMA_SERVER_EXE = env_str("LLAMA_SERVER_EXE", "llama-server.exe")
MODELS_DIR = Path(env_str("MODELS_DIR", str(PROJECT_ROOT / "models"))).expanduser()

LLAMA_CTX_SIZE = env_int("LLAMA_CTX_SIZE", 4096)
LLAMA_THREADS = env_int("LLAMA_THREADS", 6)
LLAMA_BATCH_SIZE = env_int("LLAMA_BATCH_SIZE", 512)
LLAMA_N_GPU_LAYERS = env_str("LLAMA_N_GPU_LAYERS", "0").strip()
LLAMA_EXTRA_ARGS = env_str("LLAMA_EXTRA_ARGS", "").strip()

LLAMA_STREAMING = env_bool("LLAMA_STREAMING", True)
TEXT_STREAMING = env_bool("TEXT_STREAMING", True)
LLAMA_ENABLE_AUDIO = env_bool("LLAMA_ENABLE_AUDIO", True)
LLAMA_ENABLE_IMAGES = env_bool("LLAMA_ENABLE_IMAGES", True)
LLAMA_MAX_IMAGES = env_int("LLAMA_MAX_IMAGES", 8)
LLAMA_HISTORY_TURNS = env_int("LLAMA_HISTORY_TURNS", 8)
LLAMA_STARTUP_TIMEOUT = env_float("LLAMA_STARTUP_TIMEOUT", 240)
LLAMA_REQUEST_TIMEOUT = env_float("LLAMA_REQUEST_TIMEOUT", 600)
LLAMA_REASONING_FORMAT = env_str("LLAMA_REASONING_FORMAT", "none").strip() or "none"

LLM_ENABLE_THINKING = env_str("LLM_ENABLE_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_MAX_OUTPUT_TOKENS = env_int("LLM_MAX_OUTPUT_TOKENS", 0)
DEFAULT_REPEAT_PENALTY = env_float("LLM_REPEAT_PENALTY", env_float("LLAMA_REPEAT_PENALTY", 1.18))
DEFAULT_REPEAT_LAST_N = env_int("LLM_REPEAT_LAST_N", env_int("LLAMA_REPEAT_LAST_N", 192))

AUDIO_DEBUG = env_bool("PARLOR_AUDIO_DEBUG", True)

# ── TTS ──────────────────────────────────────────────────────────
TTS_STREAMING = env_bool("TTS_STREAMING", True)
TTS_ENGINE_DEFAULT = env_str("TTS_ENGINE", "silero")

# LIVE-FIX: the .bat hardcodes TTS_EARLY_CHARS=30 / TTS_SPLIT_ON_COMMA=0,
# which causes the pause after the first words. Clamp here, bat untouched.
TTS_EARLY_CHARS = min(env_int("TTS_EARLY_CHARS", 40), 50)
TTS_LONG_CHARS = min(env_int("TTS_LONG_CHARS", 80), 90)
TTS_MAX_CHARS = min(env_int("TTS_MAX_CHARS", 160), 180)
TTS_SPLIT_ON_COMMA = True  # comma = sentence boundary in live mode, always
TTS_SENTENCE_STREAMING = env_bool("TTS_SENTENCE_STREAMING", True)

# ── STT ──────────────────────────────────────────────────────────
STT_ENGINE = env_str("STT_ENGINE", "faster_whisper").strip().lower()
STT_MODEL = env_str("STT_MODEL", "small").strip() or "small"
STT_LANG = env_str("STT_LANG", "ru").strip() or None
STT_COMPUTE_TYPE = env_str("STT_COMPUTE_TYPE", "int8").strip() or "int8"
STT_BEAM_SIZE = env_int("STT_BEAM_SIZE", 3)
STT_VAD_ENABLE = env_bool("STT_VAD_ENABLE", True)
STT_DEVICE = env_str("STT_DEVICE", "cpu").strip() or "cpu"
STT_THREADS = env_int("STT_THREADS", 2)

# ── default sampler ─────────────────────────────────────────────
DEFAULT_SAMPLER = {
    "temperature": env_float("LLM_TEMPERATURE", 1.0),
    "top_p": env_float("LLM_TOP_P", 1.0),
    "top_k": env_int("LLM_TOP_K", 0),
    "min_p": env_float("LLM_MIN_P", 0.08),
    "typical_p": env_float("LLM_TYPICAL_P", 1.0),
    "seed": env_int("LLM_SEED", 0),
    "xtc_probability": env_float("LLM_XTC_PROBABILITY", 0.0),
    "xtc_order": env_int("LLM_XTC_ORDER", 0),
    "top_n_sigma": env_float("LLM_TOP_N_SIGMA", 0.0),
    "mirostat": env_int("LLM_MIROSTAT", 0),
    "mirostat_tau": env_float("LLM_MIROSTAT_TAU", 5.0),
    "mirostat_eta": env_float("LLM_MIROSTAT_ETA", 0.1),
}

# Default system prompt — Russian first, switches to the user's language.
DEFAULT_SYSTEM_PROMPT = (
    "Ты — голосовой ИИ-ассистент. Отвечай естественно, напрямую и по текущему сообщению пользователя. "
    "Всегда учитывай предыдущие сообщения чата. "
    "Если вопрос простой — отвечай коротко; если пользователь просит объяснить, перечислить или продолжить — отвечай полно. "
    "Обычно говори по-русски, но если пользователь пишет или говорит на другом языке — отвечай на нём. "
    "Не показывай скрытые рассуждения, thought/think/reasoning-каналы, XML/служебные теги."
)

# ── Talking Head (lip-sync avatar integration) ───────────────────
TALKING_HEAD_WS = env_str("TALKING_HEAD_WS", "ws://127.0.0.1:8001")
TALKING_HEAD_ENABLED = env_bool("TALKING_HEAD_ENABLED", False)