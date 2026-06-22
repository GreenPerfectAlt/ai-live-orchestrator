"""
Parlor Jarvis v14 — llama.cpp-only local voice/text/vision chat.

The browser can attach camera/screen/PDF/video frames to each text or voice turn.
This build talks only to llama-server /v1/chat/completions.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
try:
    import tts
    TTS_SUPERTONIC_IMPORT_ERROR = None
except Exception as exc:
    tts = None
    TTS_SUPERTONIC_IMPORT_ERROR = exc
try:
    import tts_silero
    TTS_SILERO_IMPORT_ERROR = None
except Exception as exc:
    tts_silero = None
    TTS_SILERO_IMPORT_ERROR = exc
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

LLM_BACKEND = "llama_cpp"

MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).parent / "models" / "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"))
MODEL_LABEL = os.environ.get("MODEL_LABEL", Path(MODEL_PATH).name)
LAUNCHER_NAME = os.environ.get("LAUNCHER_NAME", "unknown.bat")
LLM_ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "0"))  # 0 / -1 = do not send max_tokens to llama.cpp
DEFAULT_REPEAT_PENALTY = float(os.environ.get("LLM_REPEAT_PENALTY", os.environ.get("LLAMA_REPEAT_PENALTY", "1.18")))
DEFAULT_REPEAT_LAST_N = int(os.environ.get("LLM_REPEAT_LAST_N", os.environ.get("LLAMA_REPEAT_LAST_N", "192")))

LLAMA_HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1").rstrip("/")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", os.environ.get("LLAMA_MODEL_ID", "local-gemma"))
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "no-key")
LLAMA_AUTO_START = os.environ.get("LLAMA_AUTO_START", "0").strip().lower() in {"1", "true", "yes", "on"}
LLAMA_SERVER_EXE = os.environ.get("LLAMA_SERVER_EXE", "llama-server.exe")
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(Path(__file__).parent / "models"))).expanduser()
LLAMA_CTX_SIZE = int(os.environ.get("LLAMA_CTX_SIZE", "4096"))
LLAMA_THREADS = int(os.environ.get("LLAMA_THREADS", "6"))
LLAMA_BATCH_SIZE = int(os.environ.get("LLAMA_BATCH_SIZE", "512"))
LLAMA_N_GPU_LAYERS = os.environ.get("LLAMA_N_GPU_LAYERS", "0").strip()
LLAMA_EXTRA_ARGS = os.environ.get("LLAMA_EXTRA_ARGS", "").strip()
LLAMA_STREAMING = os.environ.get("LLAMA_STREAMING", "1").strip().lower() not in {"0", "false", "no", "off"}
TEXT_STREAMING = os.environ.get("TEXT_STREAMING", "1").strip().lower() not in {"0", "false", "no", "off"}
LLAMA_ENABLE_AUDIO = os.environ.get("LLAMA_ENABLE_AUDIO", "1").strip().lower() not in {"0", "false", "no", "off"}
# Keep raw input_audio optional: Gemma 4 audio via llama.cpp is experimental and can confuse the model.
# Default: if Browser STT text exists, use that text as the user message and do not attach raw audio.
LLAMA_SEND_AUDIO_WITH_STT = os.environ.get("LLAMA_SEND_AUDIO_WITH_STT", "0").strip().lower() in {"1", "true", "yes", "on"}
AUDIO_DEBUG = os.environ.get("PARLOR_AUDIO_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}

def audio_log(event: str, **kwargs):
    if not AUDIO_DEBUG:
        return
    try:
        payload = {"event": event, **kwargs}
        print("[VOICE] " + json.dumps(payload, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f"[VOICE] log failed: {exc}", flush=True)

TTS_STREAMING = os.environ.get("TTS_STREAMING", "1").strip().lower() not in {"0", "false", "no", "off"}
TTS_EARLY_CHARS = int(os.environ.get("TTS_EARLY_CHARS", "48"))
TTS_LONG_CHARS = int(os.environ.get("TTS_LONG_CHARS", "120"))
TTS_MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "190"))
TTS_SPLIT_ON_COMMA = os.environ.get("TTS_SPLIT_ON_COMMA", "0").strip().lower() in {"1", "true", "yes", "on"}
TTS_SENTENCE_STREAMING = os.environ.get("TTS_SENTENCE_STREAMING", "1").strip().lower() in {"1", "true", "yes", "on"}
LLAMA_ENABLE_IMAGES = os.environ.get("LLAMA_ENABLE_IMAGES", "1").strip().lower() not in {"0", "false", "no", "off"}
LLAMA_MAX_IMAGES = int(os.environ.get("LLAMA_MAX_IMAGES", "8"))
LLAMA_STARTUP_TIMEOUT = float(os.environ.get("LLAMA_STARTUP_TIMEOUT", "240"))
LLAMA_REQUEST_TIMEOUT = float(os.environ.get("LLAMA_REQUEST_TIMEOUT", "600"))
LLAMA_HISTORY_TURNS = int(os.environ.get("LLAMA_HISTORY_TURNS", "8"))
LLAMA_REASONING_FORMAT = os.environ.get("LLAMA_REASONING_FORMAT", "none").strip() or "none"

DEFAULT_SAMPLER = {
    "temperature": float(os.environ.get("LLM_TEMPERATURE", "1.3")),
    "top_p": float(os.environ.get("LLM_TOP_P", "1.0")),
    "top_k": int(os.environ.get("LLM_TOP_K", "0")),
    "min_p": float(os.environ.get("LLM_MIN_P", "0.08")),
    "typical_p": float(os.environ.get("LLM_TYPICAL_P", "1.0")),
    "seed": int(os.environ.get("LLM_SEED", "0")),
}

DEFAULT_SYSTEM_PROMPT = (
    "Ты — голосовой ИИ-ассистент Parlor. Отвечай естественно, напрямую и по текущему сообщению пользователя. "
    "Всегда учитывай предыдущие сообщения чата: короткие ответы вроде «да», «ага», «нет», «дальше», "
    "«мне интересно с точки зрения химии» относятся к последней теме, а не являются новой темой сами по себе. "
    "Если вопрос простой — отвечай коротко; если пользователь просит объяснить, перечислить или продолжить — отвечай полно. "
    "Если сообщение пришло через voice/STT, считай распознанный текст обычной репликой пользователя. "
    "Не обсуждай сам факт аудио/STT и не говори, что не можешь слушать аудио, когда передан текст. "
    "Обычно говори по-русски, но если пользователь пишет или говорит по-английски — отвечай по-английски. "
    "Названия моделей, библиотек, API, путей и команд сохраняй как в оригинале. "
    "Не показывай скрытые рассуждения, thought/think/reasoning-каналы, XML/служебные теги и tool calls. "
    "Не повторяй один и тот же фрагмент ответа и не дописывай постепенно один и тот же префикс."
)

FAST_REPLY_HINT = (
    "Отвечай сразу по смыслу текущей реплики с учётом истории. "
    "Без скрытых рассуждений, без thought/think/reasoning, без tool calls и без служебных тегов."
)

CONTROL_TOKEN_RE = re.compile(r"<\|/?[^>\n]{0,80}?\|>|<\|[^>\n]{0,80}?\|>", re.IGNORECASE)
XML_CONTROL_RE = re.compile(r"</?(?:tool|tool_call|tool_response|turn|channel|assistant|model|user|system)[^>]*>", re.IGNORECASE)
THINK_PAIR_RE = re.compile(r"<(think|thought|analysis|reasoning)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE = re.compile(r"<(think|thought|analysis|reasoning)\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
CHANNEL_PAIR_RE = re.compile(r"<\|channel\>\s*(?:thought|analysis|reasoning)\s*\n.*?<channel\|>", re.IGNORECASE | re.DOTALL)
CHANNEL_OPEN_RE = re.compile(r"<\|channel\>\s*(?:thought|analysis|reasoning)\s*\n.*$", re.IGNORECASE | re.DOTALL)
LABEL_RE = re.compile(r"\b(?:Транскрипция|Ответ|Assistant|Model)\s*:\s*", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t]{2,}")
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])(?:\s+|$)|\n+")

tts_backend = None
tts_backends: dict[str, Any] = {}
tts_backend_lock = threading.Lock()
tts_loading_keys: set[str] = set()
llama_server_process: subprocess.Popen[Any] | None = None
llama_active_signature: str | None = None
llama_active_model_path: str | None = None
llama_process_lock = threading.Lock()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def normalize_tts_engine(value: Any) -> str:
    raw = str(value or os.environ.get("TTS_ENGINE", "supertonic")).strip().lower()
    if raw in {"silero", "silero_ru", "silero-ru", "ru"}:
        return "silero"
    return "supertonic"


def normalize_silero_speaker(value: Any) -> str:
    raw = str(value or os.environ.get("SILERO_SPEAKER", "baya")).strip().lower()
    mapping = {"f4": "baya", "f3": "xenia", "female": "baya", "male": "aidar"}
    raw = mapping.get(raw, raw)
    allowed = {"baya", "xenia", "kseniya", "aidar", "eugene", "random"}
    return raw if raw in allowed else "baya"


def tts_cache_key(engine: str = "supertonic", settings: dict[str, Any] | None = None) -> str:
    settings = settings or {}
    engine = normalize_tts_engine(engine)
    if engine == "silero":
        speaker = normalize_silero_speaker(settings.get("silero_speaker") or settings.get("voice"))
        speed = clamp_float(settings.get("silero_speed"), float(os.environ.get("SILERO_SPEED", os.environ.get("TTS_SPEED", "1.0"))), 0.85, 1.2)
        sample_rate = clamp_int(settings.get("silero_sample_rate"), int(os.environ.get("SILERO_SAMPLE_RATE", "24000")), 8000, 48000)
        model_id = str(settings.get("silero_model") or os.environ.get("SILERO_MODEL", "v4_ru")).strip() or "v4_ru"
        return f"silero:{model_id}:{speaker}:{sample_rate}:{speed:.3f}"
    return "supertonic"


def get_cached_tts_backend(engine: str = "supertonic", settings: dict[str, Any] | None = None):
    key = tts_cache_key(engine, settings)
    with tts_backend_lock:
        return tts_backends.get(key)


def start_tts_background_load(engine: str = "supertonic", settings: dict[str, Any] | None = None) -> str:
    settings_copy = dict(settings or {})
    engine = normalize_tts_engine(engine)
    key = tts_cache_key(engine, settings_copy)
    with tts_backend_lock:
        if key in tts_backends:
            return key
        if key in tts_loading_keys:
            return key
        tts_loading_keys.add(key)

    def _load():
        try:
            print(f"🔊 Background TTS load started: {key}")
            get_tts_backend(engine, settings_copy)
            print(f"✅ Background TTS ready: {key}")
        except Exception as exc:
            print(f"⚠️ Background TTS load failed ({key}): {exc}")
        finally:
            with tts_backend_lock:
                tts_loading_keys.discard(key)

    threading.Thread(target=_load, daemon=True).start()
    return key


def get_tts_backend(engine: str = "supertonic", settings: dict[str, Any] | None = None):
    """Lazy TTS loader. Supertonic stays as existing tts.py; Silero is optional."""
    global tts_backend
    settings = settings or {}
    engine = normalize_tts_engine(engine)

    if engine == "silero":
        if tts_silero is None:
            raise RuntimeError(f"Silero backend is unavailable: {TTS_SILERO_IMPORT_ERROR}")
        speaker = normalize_silero_speaker(settings.get("silero_speaker") or settings.get("voice"))
        speed = clamp_float(settings.get("silero_speed"), float(os.environ.get("SILERO_SPEED", os.environ.get("TTS_SPEED", "1.0"))), 0.85, 1.2)
        sample_rate = clamp_int(settings.get("silero_sample_rate"), int(os.environ.get("SILERO_SAMPLE_RATE", "24000")), 8000, 48000)
        model_id = str(settings.get("silero_model") or os.environ.get("SILERO_MODEL", "v4_ru")).strip() or "v4_ru"
        key = f"silero:{model_id}:{speaker}:{sample_rate}:{speed:.3f}"
        with tts_backend_lock:
            backend = tts_backends.get(key)
            if backend is None:
                print(f"🔊 Loading TTS backend: Silero RU | speaker={speaker}, speed={speed}, sr={sample_rate}")
                backend = tts_silero.load(model_id=model_id, speaker=speaker, sample_rate=sample_rate, speed=speed)
                tts_backends[key] = backend
            return backend

    # Existing Supertonic backend from user's tts.py. Loaded once from env.
    if tts is None:
        raise RuntimeError(f"Supertonic backend is unavailable: {TTS_SUPERTONIC_IMPORT_ERROR}")
    with tts_backend_lock:
        backend = tts_backends.get("supertonic")
        if backend is None:
            print("🔊 Loading TTS backend: Supertonic 3")
            backend = tts.load()
            tts_backends["supertonic"] = backend
            tts_backend = backend
        return backend


def normalize_sampler(settings: dict[str, Any] | None) -> dict[str, Any]:
    settings = settings or {}
    # max_output_tokens <= 0 means: do not send max_tokens to llama.cpp, so server-side -n -1 can work.
    raw_max = settings.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    try:
        max_out = int(raw_max)
    except (TypeError, ValueError):
        max_out = DEFAULT_MAX_OUTPUT_TOKENS
    if max_out > 0:
        max_out = max(32, min(32768, max_out))
    return {
        "temperature": clamp_float(settings.get("temperature"), DEFAULT_SAMPLER["temperature"], 0.0, 2.0),
        "top_p": clamp_float(settings.get("top_p"), DEFAULT_SAMPLER["top_p"], 0.0, 1.0),
        "top_k": clamp_int(settings.get("top_k"), DEFAULT_SAMPLER["top_k"], 0, 256),
        "min_p": clamp_float(settings.get("min_p"), DEFAULT_SAMPLER["min_p"], 0.0, 1.0),
        "typical_p": clamp_float(settings.get("typical_p"), DEFAULT_SAMPLER["typical_p"], 0.0, 1.0),
        "seed": clamp_int(settings.get("seed"), DEFAULT_SAMPLER["seed"], 0, 2_147_483_647),
        "max_output_tokens": max_out,
        "enable_thinking": bool(settings.get("enable_thinking", LLM_ENABLE_THINKING)),
        "repeat_penalty": clamp_float(settings.get("repeat_penalty"), DEFAULT_REPEAT_PENALTY, 1.0, 2.0),
        "repeat_last_n": clamp_int(settings.get("repeat_last_n"), DEFAULT_REPEAT_LAST_N, 0, 32768),
    }


def strip_thinking_and_controls(text: str, *, final: bool = False) -> str:
    if not text:
        return ""
    text = CHANNEL_PAIR_RE.sub("", text)
    text = THINK_PAIR_RE.sub("", text)
    if not final:
        text = CHANNEL_OPEN_RE.sub("", text)
        text = THINK_OPEN_RE.sub("", text)
    text = CONTROL_TOKEN_RE.sub("", text)
    text = XML_CONTROL_RE.sub("", text)
    text = LABEL_RE.sub("", text)
    text = text.replace("<channel|>", "").replace("<tool|>", "").replace("<turn|>", "")
    # Extremely common broken overlap from old cumulative streaming. Keep it generic and harmless.
    text = re.sub(r"\bБото(?=Ботокс\b)", "", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip() if final else text


def sanitize_tts_text(text: str) -> str:
    """Text shown in chat may contain emoji/markdown-ish symbols; local TTS often
    pauses or stumbles on them. Keep chat text unchanged, but speak a cleaner string.
    """
    text = strip_thinking_and_controls(text or "", final=True)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+", "", text)
    text = text.replace("*", "").replace("_", "").replace("`", "")
    text = re.sub(r"\s+([.!?…])", r"\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text



def collapse_generated_repeats(text: str) -> str:
    """Best-effort cleanup for broken cumulative token streams.

    The real fix is non-stream generation. This is just a safety net for already
    corrupted text such as: "ПриветПривет. Я Я голосо Я голосовой ...".
    """
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"([.!?…])\s*\1+", r"\1", text)
    # duplicated long words without a separator: ПриветПривет -> Привет
    text = re.sub(r"\b([A-Za-zА-Яа-яЁё]{3,})(?:\1\b)+", r"\1", text)
    # adjacent duplicate words: Я Я -> Я
    text = re.sub(r"\b([\wА-Яа-яЁё-]{2,})(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

    def norm_words(words: list[str]) -> str:
        return re.sub(r"\W+", "", " ".join(words).casefold())

    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        handled = False
        max_m = min(8, len(out), len(words) - i)
        for m in range(max_m, 0, -1):
            old = norm_words(out[-m:])
            new = norm_words(words[i:i + m])
            if not old or not new:
                continue
            if old == new:
                i += m
                handled = True
                break
            # Previous phrase is an unfinished prefix of the new phrase.
            if new.startswith(old) and len(new) > len(old) + 1 and len(old) >= 4:
                del out[-m:]
                handled = True
                break
            # Current phrase is an unfinished duplicate of the previous phrase.
            if old.startswith(new) and len(old) > len(new) + 1 and len(new) >= 4:
                i += m
                handled = True
                break
        if handled:
            continue
        if out:
            prev = norm_words([out[-1]])
            cur = norm_words([words[i]])
            if prev == cur:
                i += 1
                continue
            if cur.startswith(prev) and len(cur) > len(prev) + 1 and len(prev) >= 4:
                out.pop()
            elif prev.startswith(cur) and len(prev) > len(cur) + 1 and len(cur) >= 4:
                i += 1
                continue
        out.append(words[i])
        i += 1
    text = " ".join(out)

    # Drop exactly repeated consecutive sentences.
    parts = re.split(r"(?<=[.!?…])\s+", text)
    cleaned: list[str] = []
    last_key = ""
    for part in parts:
        key = re.sub(r"\W+", "", part.casefold())
        if key and key == last_key:
            continue
        cleaned.append(part)
        if key:
            last_key = key
    text = " ".join(cleaned)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def clean_generated_response(text: str) -> str:
    return collapse_generated_repeats(strip_thinking_and_controls(text, final=True))


def final_tts_sentences(text: str) -> list[str]:
    complete, tail = extract_sentences(text)
    if tail.strip():
        complete.append(tail.strip())
    out: list[str] = []
    seen: set[str] = set()
    for sentence in complete:
        clean = clean_generated_response(sentence)
        key = re.sub(r"\W+", "", clean.casefold())[:240]
        if len(clean) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out

def text_from_chunk(chunk: Any) -> str:
    if isinstance(chunk, dict):
        parts = chunk.get("content") or []
        if isinstance(parts, list):
            return "".join(str(item.get("text", "")) for item in parts if isinstance(item, dict) and item.get("type", "text") == "text")
        return str(chunk.get("text", ""))
    return str(getattr(chunk, "text", "") or "")


def normalize_stream_delta(chunk_text: str, emitted_text: str) -> tuple[str, str]:
    text = chunk_text or ""
    if not text:
        return "", emitted_text
    if text.startswith(emitted_text):
        return text[len(emitted_text):], text
    if emitted_text.endswith(text):
        return "", emitted_text
    max_overlap = min(len(emitted_text), len(text))
    for n in range(max_overlap, 0, -1):
        if emitted_text.endswith(text[:n]):
            delta = text[n:]
            return delta, emitted_text + delta
    return text, emitted_text + text


def extract_sentences(buffer: str) -> tuple[list[str], str]:
    complete: list[str] = []
    start = 0
    for match in SENTENCE_END_RE.finditer(buffer):
        end = match.end()
        sentence = buffer[start:end].strip()
        if sentence:
            complete.append(sentence)
        start = end
    return complete, buffer[start:]


def extract_speak_chunks(buffer: str, *, force: bool = False, first: bool = False) -> tuple[list[str], str]:
    """v45: speak completed sentences immediately.

    If the model has already produced a sentence ending (. ! ? … or newline),
    send that sentence to TTS right away. This prevents waiting for the whole
    answer or waiting for large character thresholds.
    """
    buf = SPACE_RE.sub(" ", (buffer or "").strip())
    if not buf:
        return [], ""

    if TTS_SENTENCE_STREAMING:
        complete, tail = extract_sentences(buf)
        if complete:
            min_sentence = 14 if first else 24
            out: list[str] = []
            carry = ""
            for sentence in complete:
                sentence = SPACE_RE.sub(" ", sentence.strip())
                if not sentence:
                    continue
                if carry:
                    sentence = (carry + " " + sentence).strip()
                    carry = ""
                if len(sentence) < min_sentence and not force:
                    carry = sentence
                    continue
                clean = sanitize_tts_text(sentence)
                if clean:
                    out.append(clean)
            if out:
                rest_parts = []
                if carry:
                    rest_parts.append(carry)
                if tail.strip():
                    rest_parts.append(tail.strip())
                return out, " ".join(rest_parts).strip()

    # Fallback for long text without punctuation.
    first_chars = max(48, int(TTS_EARLY_CHARS))
    target_chars = max(first_chars + 48, int(TTS_LONG_CHARS))
    max_chars = max(target_chars + 48, int(TTS_MAX_CHARS))
    out: list[str] = []
    threshold = first_chars if first else target_chars
    min_sentence = 18 if first else 40

    while len(buf) >= threshold:
        window_len = min(len(buf), max_chars)
        window = buf[:window_len]
        split_at = -1

        sentence_ends = [m.end() for m in SENTENCE_END_RE.finditer(window) if m.end() >= min_sentence]
        if sentence_ends:
            split_at = sentence_ends[0]

        if split_at < 0 and TTS_SPLIT_ON_COMMA:
            for sep in [", ", "; ", ": ", " — ", " - "]:
                idx = window.rfind(sep, threshold, window_len)
                if idx >= threshold:
                    split_at = idx + len(sep)
                    break

        if split_at < 0:
            idx = window.rfind(" ", threshold, window_len)
            if idx >= threshold:
                split_at = idx + 1

        if split_at < 0:
            break

        chunk = sanitize_tts_text(buf[:split_at])
        if chunk:
            out.append(chunk)
        buf = buf[split_at:].lstrip()
        threshold = target_chars
        min_sentence = 40
        first = False

    if force and buf:
        chunk = sanitize_tts_text(buf)
        if chunk:
            out.append(chunk)
        buf = ""

    return out, buf

def stable_prompt_id(prompt: str, sampler: dict[str, Any] | None = None) -> str:
    payload = json.dumps({"prompt": prompt, "sampler": sampler or {}}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {LLAMA_API_KEY}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8", "replace")
    return json.loads(data) if data else {}


def wait_for_llama_server() -> None:
    deadline = time.time() + LLAMA_STARTUP_TIMEOUT
    last_error: Exception | None = None
    url = f"{LLAMA_BASE_URL}/models"
    while time.time() < deadline:
        try:
            info = http_json("GET", url, timeout=5.0)
            model_ids = [str(item.get("id", "")) for item in info.get("data", []) if isinstance(item, dict)]
            print(f"✅ llama.cpp server ready: {LLAMA_BASE_URL} | models={model_ids or 'unknown'}")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"llama.cpp server is not reachable at {LLAMA_BASE_URL}. "
        f"Start llama-server first. Last error: {last_error}"
    )



def resolve_model_file(value: Any) -> Path | None:
    """Resolve a model/mmproj path. Relative paths are resolved inside MODELS_DIR."""
    text = str(value or "").strip().strip('"')
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = MODELS_DIR / path
    return path.resolve()


def llama_model_key(path: Path) -> str:
    name = path.name.lower()
    m = re.search(r"gemma[-_]?4[-_]?e([24])b[-_]?it[-_]?qat", name, re.IGNORECASE)
    if m:
        return f"gemma-4-e{m.group(1)}b-it-qat"
    # Fallback: strip quantization suffixes so mmproj can still be paired by common prefix.
    stem = path.stem.lower()
    stem = re.sub(r"[-_]?ud[-_]?q\d.*$", "", stem)
    stem = re.sub(r"[-_]?q\d.*$", "", stem)
    return stem


def scan_llama_models() -> list[dict[str, Any]]:
    base = MODELS_DIR.resolve()
    if not base.exists():
        return []
    all_files = [p for p in base.rglob("*") if p.is_file()]
    mmprojs = [p for p in all_files if "mmproj" in p.name.lower()]
    models = [p for p in all_files if p.suffix.lower() == ".gguf" and "mmproj" not in p.name.lower()]
    items: list[dict[str, Any]] = []
    for model in sorted(models, key=lambda p: p.name.lower()):
        key = llama_model_key(model)
        paired: Path | None = None
        for mm in mmprojs:
            if key and key in mm.name.lower():
                paired = mm
                break
        if paired is None:
            # fallback by E2B/E4B token
            token = "e2b" if "e2b" in model.name.lower() else "e4b" if "e4b" in model.name.lower() else ""
            if token:
                paired = next((mm for mm in mmprojs if token in mm.name.lower()), None)
        rel_model = str(model.relative_to(base)) if model.is_relative_to(base) else str(model)
        rel_mm = str(paired.relative_to(base)) if paired and paired.is_relative_to(base) else (str(paired) if paired else "")
        items.append({
            "id": hashlib.sha1(str(model).encode("utf-8", "ignore")).hexdigest()[:12],
            "name": model.name,
            "label": model.stem,
            "path": rel_model,
            "absolute_path": str(model),
            "mmproj_name": paired.name if paired else "",
            "mmproj_path": rel_mm,
            "mmproj_absolute_path": str(paired) if paired else "",
        })
    return items


def server_url_is_ready(timeout: float = 2.0) -> bool:
    try:
        http_json("GET", f"{LLAMA_BASE_URL}/models", timeout=timeout)
        return True
    except Exception:
        return False


def stop_managed_llama_server() -> None:
    global llama_server_process, llama_active_signature, llama_active_model_path
    proc = llama_server_process
    llama_server_process = None
    llama_active_signature = None
    llama_active_model_path = None
    if proc and proc.poll() is None:
        print("🛑 stopping previous llama-server...")
        with suppress(Exception):
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            with suppress(Exception):
                proc.kill()


def pick_default_llama_model() -> tuple[str, str]:
    models = scan_llama_models()
    if not models:
        raise RuntimeError(f"No GGUF models found in {MODELS_DIR}. Put .gguf files into the models folder or select a path in the UI.")
    # Prefer E2B for fast realtime voice, then first sorted model.
    chosen = next((m for m in models if "e2b" in m["name"].lower()), models[0])
    return str(chosen.get("path") or ""), str(chosen.get("mmproj_path") or "")


def ensure_llama_model(model_path_value: Any = None, mmproj_path_value: Any = None) -> dict[str, Any]:
    """Start/switch managed llama-server when LLAMA_AUTO_START=1, otherwise only wait for external server."""
    global llama_server_process, llama_active_signature, llama_active_model_path, LLAMA_MODEL
    if not LLAMA_AUTO_START:
        wait_for_llama_server()
        return {"backend": "llama_cpp", "managed": False, "base_url": LLAMA_BASE_URL, "model": LLAMA_MODEL}

    if not str(model_path_value or "").strip():
        model_path_value, mmproj_path_value = pick_default_llama_model()

    model_path = resolve_model_file(model_path_value)
    mmproj_path = resolve_model_file(mmproj_path_value) if str(mmproj_path_value or "").strip() else None
    if model_path is None or not model_path.exists():
        raise RuntimeError(f"Selected GGUF model not found: {model_path_value!r}")
    if mmproj_path is not None and not mmproj_path.exists():
        print(f"⚠️ mmproj not found, starting without it: {mmproj_path}")
        mmproj_path = None

    signature = hashlib.sha1(json.dumps({
        "model": str(model_path),
        "mmproj": str(mmproj_path or ""),
        "host": LLAMA_HOST,
        "port": LLAMA_PORT,
        "ctx": LLAMA_CTX_SIZE,
        "threads": LLAMA_THREADS,
        "batch": LLAMA_BATCH_SIZE,
        "ngl": LLAMA_N_GPU_LAYERS,
        "extra": LLAMA_EXTRA_ARGS,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    with llama_process_lock:
        if llama_active_signature == signature and server_url_is_ready(timeout=2.0):
            return {"backend": "llama_cpp", "managed": True, "base_url": LLAMA_BASE_URL, "model": model_path.name, "mmproj": mmproj_path.name if mmproj_path else ""}

        stop_managed_llama_server()

        exe = LLAMA_SERVER_EXE.strip().strip('"') or "llama-server.exe"
        exe_path = Path(exe).expanduser()
        exe_cmd = str(exe_path) if exe_path.exists() else (shutil.which(exe) or exe)
        cmd = [exe_cmd, "-m", str(model_path), "--host", LLAMA_HOST, "--port", str(LLAMA_PORT), "--ctx-size", str(LLAMA_CTX_SIZE), "--threads", str(LLAMA_THREADS), "--batch-size", str(LLAMA_BATCH_SIZE)]
        if LLAMA_N_GPU_LAYERS:
            cmd += ["-ngl", LLAMA_N_GPU_LAYERS]
        if mmproj_path:
            cmd += ["--mmproj", str(mmproj_path)]
        if LLAMA_EXTRA_ARGS:
            cmd += shlex.split(LLAMA_EXTRA_ARGS)

        print("🚀 launching llama-server:")
        print("   " + " ".join(f'\"{c}\"' if " " in c else c for c in cmd))
        try:
            llama_server_process = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
        except FileNotFoundError as exc:
            raise RuntimeError(f"llama-server.exe not found. Put it near run_llama.bat or set LLAMA_SERVER_EXE. Tried: {exe_cmd}") from exc
        LLAMA_MODEL = model_path.stem
        wait_for_llama_server()
        llama_active_signature = signature
        llama_active_model_path = str(model_path)
        return {"backend": "llama_cpp", "managed": True, "base_url": LLAMA_BASE_URL, "model": model_path.name, "mmproj": mmproj_path.name if mmproj_path else ""}


def load_models() -> None:
    global tts_backend
    print("🧠 LLM backend: llama.cpp only")
    print(f"🚀 llama.cpp: {LLAMA_BASE_URL} | model={LLAMA_MODEL}")
    print(f"⚙️ llama streaming={LLAMA_STREAMING}, tts_streaming={TTS_STREAMING}, audio={LLAMA_ENABLE_AUDIO}, send_audio_with_stt={LLAMA_SEND_AUDIO_WITH_STT}, images={LLAMA_ENABLE_IMAGES}, auto_start={LLAMA_AUTO_START}")
    print(f"📁 models dir: {MODELS_DIR}")
    if not LLAMA_AUTO_START:
        wait_for_llama_server()
    else:
        found = scan_llama_models()
        print(f"🔎 GGUF models found: {len(found)}. Model is selected by the current .bat; server will start on first llama request.")

    print("🔊 TTS: Supertonic 3 + Silero RU")
    # Do not block app/model responses on TTS loading. Silero/PyTorch/hub can take
    # a long time on first run, so load it in the background.
    if env_bool("TTS_BACKGROUND_PRELOAD", True):
        start_tts_background_load(os.environ.get("TTS_ENGINE", "silero"), {
            "silero_speaker": os.environ.get("SILERO_SPEAKER", "baya"),
            "silero_speed": os.environ.get("SILERO_SPEED", "1.0"),
            "silero_sample_rate": os.environ.get("SILERO_SAMPLE_RATE", "24000"),
            "silero_model": os.environ.get("SILERO_MODEL", "v4_ru"),
        })
    elif env_bool("TTS_PRELOAD", False):
        try:
            get_tts_backend(os.environ.get("TTS_ENGINE", "supertonic"), {})
        except Exception as exc:
            print(f"⚠️ TTS preload failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text(encoding="utf-8"))



@app.get("/api/llama/models")
async def api_llama_models():
    return {
        "backend": LLM_BACKEND,
        "auto_start": LLAMA_AUTO_START,
        "models_dir": str(MODELS_DIR),
        "base_url": LLAMA_BASE_URL,
        "active_model_path": llama_active_model_path or "",
        "models": scan_llama_models(),
    }


@app.post("/api/llama/select")
async def api_llama_select(payload: dict[str, Any]):
    info = ensure_llama_model(payload.get("model_path"), payload.get("mmproj_path"))
    return {"ok": True, **info}


@dataclass
class LlamaSession:
    chat_id: str
    prompt_id: str
    history: list[dict[str, Any]] = field(default_factory=list)



def extract_image_infos(msg: dict[str, Any], limit: int | None = None) -> list[dict[str, str]]:
    """Return image/frame blobs in display order.

    Frontend v10 can send:
      - image: one legacy frame
      - images: still images from camera/screen/pdf/video
      - frames: time-ordered live samples from screen/camera/video
    llama.cpp's OpenAI endpoint receives them as several image_url parts.
    """
    if limit is None:
        limit = max(1, LLAMA_MAX_IMAGES)
    infos: list[dict[str, str]] = []

    def add_item(item: Any, default_source: str = "image") -> None:
        if len(infos) >= limit:
            return
        source = default_source
        blob: Any = None
        ts = ""
        idx = ""
        kind = ""
        if isinstance(item, dict):
            blob = item.get("blob") or item.get("image") or item.get("data")
            source = str(item.get("source") or item.get("name") or source).strip().lower()[:40] or source
            if item.get("t") is not None:
                ts = str(item.get("t"))[:32]
            if item.get("index") is not None:
                idx = str(item.get("index"))[:16]
            if item.get("kind") is not None:
                kind = str(item.get("kind"))[:32]
        else:
            blob = item
        if isinstance(blob, str) and blob.strip():
            info = {"source": source, "blob": blob.strip()}
            if ts:
                info["t"] = ts
            if idx:
                info["index"] = idx
            if kind:
                info["kind"] = kind
            infos.append(info)

    direct = msg.get("image")
    if isinstance(direct, str) and direct.strip():
        add_item({"source": "image", "blob": direct.strip(), "kind": "legacy"})

    # Stills first, then real-time frame buffers. The frontend normally sends
    # only frames in v10, but keep both for compatibility.
    raw_images = msg.get("images")
    if isinstance(raw_images, list):
        for item in raw_images:
            add_item(item, "image")
            if len(infos) >= limit:
                break

    raw_frames = msg.get("frames")
    if isinstance(raw_frames, list):
        # Preserve order from browser: oldest -> newest within each source.
        for item in raw_frames:
            add_item(item, "frame")
            if len(infos) >= limit:
                break

    return infos[:limit]

def extract_image_blobs(msg: dict[str, Any], limit: int = 4) -> list[str]:
    return [x["blob"] for x in extract_image_infos(msg, limit=limit)]

def data_uri_from_base64(data: str, mime: str) -> str:
    data = (data or "").strip()
    if data.startswith("data:"):
        return data
    return f"data:{mime};base64,{data}"


def make_llama_user_content(msg: dict[str, Any], user_text: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    has_audio = bool(msg.get("audio"))
    image_infos = extract_image_infos(msg)
    has_image = bool(image_infos)
    sources = [info.get("source", "image") for info in image_infos]

    source_note = ""
    if sources:
        source_note = "\n\n[Визуальные вложения по порядку кадров: "
        source_note += ", ".join(f"{i + 1}) {src}" for i, src in enumerate(sources)) + ". "
        if "screen" in sources:
            source_note += "screen = запись экрана пользователя; сравнивай кадры по порядку. "
        if "video" in sources:
            source_note += "video = кадры из загруженного видеофайла. "
        if "camera" in sources:
            source_note += "camera = камера пользователя. "
        if "pdf" in sources:
            source_note += "pdf = текущая страница документа. "
        source_note += "]"

    # Important: keep the user turn clean. Small local Gemma models often start
    # answering helper wrappers like "this is an instruction" if we append per-turn
    # meta-prompts. The behavior rules live in the system prompt; the user content
    # should remain the user's actual text plus short visual labels only.
    if user_text:
        prompt_text = user_text.strip() + source_note
    elif has_audio:
        prompt_text = (
            "Пользователь отправил голосовое сообщение как native audio/input_audio. "
            "Прослушай WAV напрямую и ответь на услышанную речь. "
            "Если пользователь просит транскрипцию — расшифруй дословно. "
            "Если речь неразборчива — коротко попроси повторить."
            f"{source_note}"
        )
    elif has_image:
        prompt_text = f"Посмотри на изображения/кадры пользователя и ответь на его запрос.{source_note}"
    else:
        prompt_text = "Продолжи разговор по последней реплике пользователя с учётом истории."
    parts.append({"type": "text", "text": prompt_text})

    audio_mode = str(
        msg.get("audio_mode")
        or (msg.get("settings") or {}).get("audio_input_mode")
        or ""
    ).strip().lower()
    attach_reason = "none"
    should_attach_audio = False
    if has_audio and LLAMA_ENABLE_AUDIO:
        if audio_mode in {"native", "hybrid", "audio", "wav"}:
            should_attach_audio = True
            attach_reason = f"audio_mode={audio_mode}"
        elif LLAMA_SEND_AUDIO_WITH_STT:
            should_attach_audio = True
            attach_reason = "LLAMA_SEND_AUDIO_WITH_STT=1"
        elif not user_text:
            should_attach_audio = True
            attach_reason = "no_user_text_raw_fallback"
        else:
            attach_reason = "stt_text_preferred"
    elif has_audio and not LLAMA_ENABLE_AUDIO:
        attach_reason = "LLAMA_ENABLE_AUDIO=0"

    audio_log(
        "llama_audio_attach_decision",
        request_id=str(msg.get("request_id") or ""),
        audio_mode=audio_mode or "stt",
        has_audio=has_audio,
        user_text_len=len(user_text or ""),
        attach_input_audio=bool(should_attach_audio),
        reason=attach_reason,
        audio_b64_len=len(str(msg.get("audio") or "")),
        audio_seconds=msg.get("audio_seconds"),
    )

    if should_attach_audio:
        parts.append({
            "type": "input_audio",
            "input_audio": {
                "data": str(msg.get("audio") or ""),
                "format": "wav",
            },
        })

    if has_image and LLAMA_ENABLE_IMAGES:
        for i, info in enumerate(image_infos, start=1):
            src = info.get("source", "image")
            t = info.get("t")
            kind = info.get("kind")
            label = f"\n[Кадр {i}: source={src}"
            if kind:
                label += f", kind={kind}"
            if t:
                label += f", t={t}"
            label += "]"
            parts.append({"type": "text", "text": label})
            parts.append({
                "type": "image_url",
                "image_url": {"url": data_uri_from_base64(info["blob"], "image/jpeg")},
            })

    return parts

def normalize_client_history(msg: dict[str, Any]) -> list[dict[str, str]]:
    raw = msg.get("history")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if text in {"…", "...", "with audio", "with audio · screen", "with audio · camera", "with audio · video"}:
            continue
        if text.startswith("[ERROR]") or text.startswith("[LLM error"):
            continue
        if role == "user" and "voice message" in text.lower():
            continue
        text = text.replace("\x00", " ")
        text = SPACE_RE.sub(" ", text)[:3000 if role == "user" else 4000]
        if text:
            out.append({"role": role, "content": text})
    max_messages = max(2, LLAMA_HISTORY_TURNS * 2)
    return out[-max_messages:]


def llama_system_prompt(system_prompt: str) -> str:
    base = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    memory_rule = (
        "\n\nВажно: учитывай историю сообщений выше. "
        "Если пользователь отвечает коротко («да», «ага», «нет», «дальше»), "
        "понимай это как продолжение последнего вопроса/темы."
    )
    if "учитывай историю" not in base.lower() and "предыдущие сообщения" not in base.lower():
        return base + memory_rule
    return base


def build_llama_messages(session: LlamaSession, system_prompt: str, msg: dict[str, Any], user_text: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": llama_system_prompt(system_prompt)}]
    client_history = normalize_client_history(msg)
    if client_history:
        messages.extend(client_history)
    elif LLAMA_HISTORY_TURNS > 0:
        messages.extend(session.history[-LLAMA_HISTORY_TURNS * 2:])
    messages.append({"role": "user", "content": make_llama_user_content(msg, user_text)})
    return messages


def llama_payload(messages: list[dict[str, Any]], sampler: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "temperature": float(sampler["temperature"]),
        "top_p": float(sampler["top_p"]),
        "top_k": int(sampler["top_k"]),
        "min_p": float(sampler.get("min_p", DEFAULT_SAMPLER.get("min_p", 0.08))),
        "typical_p": float(sampler.get("typical_p", DEFAULT_SAMPLER.get("typical_p", 1.0))),
        "repeat_penalty": float(sampler.get("repeat_penalty", DEFAULT_REPEAT_PENALTY)),
        "repeat_last_n": int(sampler.get("repeat_last_n", DEFAULT_REPEAT_LAST_N)),
        "frequency_penalty": 0.0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": bool(sampler.get("enable_thinking", False))},
        "reasoning_format": LLAMA_REASONING_FORMAT,
    }
    max_out = int(sampler.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    if max_out > 0:
        payload["max_tokens"] = max_out
    if int(sampler.get("seed", 0)) > 0:
        payload["seed"] = int(sampler["seed"])
    return payload


def extract_llama_text(obj: dict[str, Any]) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    text = choice.get("text")
    return str(text or "")


def llama_chat_once(messages: list[dict[str, Any]], sampler: dict[str, Any]) -> str:
    payload = llama_payload(messages, sampler, stream=False)
    try:
        obj = http_json("POST", f"{LLAMA_BASE_URL}/chat/completions", payload, timeout=LLAMA_REQUEST_TIMEOUT)
        return extract_llama_text(obj)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"llama.cpp HTTP {exc.code}: {body[:1000]}") from exc
    except (urllib.error.URLError, ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError) as exc:
        raise RuntimeError("llama-server оборвал соединение/упал. Для Gemma 4 mmproj проверь батник: LLAMA_UBATCH_SIZE должен быть 4096 и картинки лучше 512-640px. Исходная ошибка: " + str(exc)) from exc


def llama_chat_stream(messages: list[dict[str, Any]], sampler: dict[str, Any]) -> Iterator[str]:
    payload = llama_payload(messages, sampler, stream=True)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{LLAMA_BASE_URL}/chat/completions", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {LLAMA_API_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=LLAMA_REQUEST_TIMEOUT) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                text = ""
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
                if not text:
                    text = extract_llama_text(obj)
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"llama.cpp HTTP {exc.code}: {body_text[:1000]}") from exc


def append_llama_history(session: LlamaSession, msg: dict[str, Any], user_text: str, assistant_text: str) -> None:
    if not assistant_text.strip():
        return
    if user_text.strip():
        user_memory = user_text.strip()
    elif msg.get("audio") and (msg.get("transcription") or msg.get("transcript")):
        user_memory = "[Голосом] " + str(msg.get("transcription") or msg.get("transcript") or "").strip()[:2000]
    elif msg.get("audio") and msg.get("image"):
        user_memory = "[Пользователь отправил голосовое сообщение и изображение.]"
    elif msg.get("audio"):
        user_memory = "[Пользователь отправил голосовое сообщение.]"
    elif extract_image_blobs(msg):
        user_memory = "[Пользователь отправил изображение/кадры экрана/видео.]"
    else:
        user_memory = "[Пользователь отправил пустое сообщение.]"
    session.history.append({"role": "user", "content": user_memory[:2000]})
    session.history.append({"role": "assistant", "content": assistant_text.strip()[:4000]})
    max_messages = max(2, LLAMA_HISTORY_TURNS * 2)
    if len(session.history) > max_messages:
        del session.history[:-max_messages]


@app.get("/api/status")
async def api_status():
    return {
        "backend": LLM_BACKEND,
        "model_label": MODEL_LABEL,
        "model": LLAMA_MODEL,
        "launcher_name": LAUNCHER_NAME,
        "text_streaming": TEXT_STREAMING,
        "llama_streaming": LLAMA_STREAMING,
        "tts_streaming": TTS_STREAMING,
        "tts_early_chars": TTS_EARLY_CHARS,
        "tts_long_chars": TTS_LONG_CHARS,
        "tts_max_chars": TTS_MAX_CHARS,
        "tts_split_on_comma": TTS_SPLIT_ON_COMMA,
        "tts_engine": os.environ.get("TTS_ENGINE", "silero"),
        "send_audio_with_stt": LLAMA_SEND_AUDIO_WITH_STT,
        "max_images": LLAMA_MAX_IMAGES,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    with suppress(Exception):
        await ws.send_text(json.dumps({
            "type": "app_status",
            "backend": LLM_BACKEND,
            "model_label": MODEL_LABEL,
            "model": LLAMA_MODEL,
            "launcher_name": LAUNCHER_NAME,
            "text_streaming": TEXT_STREAMING,
            "llama_streaming": LLAMA_STREAMING,
            "tts_streaming": TTS_STREAMING,
            "send_audio_with_stt": LLAMA_SEND_AUDIO_WITH_STT,
            "max_images": LLAMA_MAX_IMAGES,
        }, ensure_ascii=False))

    interrupted = asyncio.Event()
    cancelled_requests: set[str] = set()
    msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    llama_sessions: dict[str, LlamaSession] = {}

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "ping":
                    with suppress(Exception):
                        await ws.send_text(json.dumps({"type": "pong", "t": msg.get("t")}, ensure_ascii=False))
                    continue
                if msg.get("type") == "interrupt":
                    rid = str(msg.get("request_id") or "").strip()
                    if rid:
                        cancelled_requests.add(rid)
                    interrupted.set()
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())
    loop = asyncio.get_running_loop()

    def get_llama_session(chat_id: str, system_prompt: str) -> LlamaSession:
        prompt_id = stable_prompt_id(system_prompt, {"backend": "llama_cpp"})
        existing = llama_sessions.get(chat_id)
        if existing and existing.prompt_id == prompt_id:
            return existing
        session = LlamaSession(chat_id=chat_id, prompt_id=prompt_id)
        llama_sessions[chat_id] = session
        return session

    def request_cancelled(request_id: str) -> bool:
        return bool(request_id and request_id in cancelled_requests) or interrupted.is_set()


    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break
            interrupted.clear()

            if str(msg.get("type") or "").lower() == "tts":
                request_id = str(msg.get("request_id") or f"tts-{int(time.time()*1000)}")
                replay_text = strip_thinking_and_controls(str(msg.get("text") or ""), final=True).strip()
                if replay_text:
                    audio_started = False
                    tts_total_time = 0.0
                    sentence_index = 0
                    replay_settings = {
                        "voice": msg.get("voice"),
                        "silero_speaker": msg.get("silero_speaker"),
                        "silero_speed": msg.get("silero_speed"),
                    }
                    replay_engine = normalize_tts_engine(msg.get("tts_engine"))
                    replay_backend = get_cached_tts_backend(replay_engine, replay_settings)
                    if replay_backend is None:
                        start_tts_background_load(replay_engine, replay_settings)
                    try:
                        if replay_backend is None:
                            replay_backend = await loop.run_in_executor(None, lambda: get_tts_backend(replay_engine, replay_settings))
                    except Exception as exc:
                        err = f"TTS replay backend error: {exc}"
                        print(err)
                        with suppress(Exception):
                            await ws.send_text(json.dumps({"type": "error", "request_id": request_id, "message": err}, ensure_ascii=False))
                        continue
                    chunks = final_tts_sentences(replay_text) or [replay_text]
                    for sentence in chunks:
                        if request_cancelled(request_id):
                            break
                        clean_sentence = sanitize_tts_text(sentence).strip()
                        if len(clean_sentence) < 2:
                            continue
                        if not audio_started:
                            await ws.send_text(json.dumps({
                                "type": "audio_start",
                                "request_id": request_id,
                                "sample_rate": replay_backend.sample_rate,
                            }))
                            audio_started = True
                        tts0 = time.time()
                        pcm = await loop.run_in_executor(None, lambda s=clean_sentence: replay_backend.generate(s))
                        tts_total_time += time.time() - tts0
                        if request_cancelled(request_id):
                            break
                        pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                        await ws.send_text(json.dumps({
                            "type": "audio_chunk",
                            "request_id": request_id,
                            "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                            "index": sentence_index,
                        }))
                        sentence_index += 1
                    if audio_started and not request_cancelled(request_id):
                        await ws.send_text(json.dumps({
                            "type": "audio_end",
                            "request_id": request_id,
                            "tts_time": round(tts_total_time, 2),
                        }))
                continue

            chat_id = str(msg.get("chat_id") or "default")[:80]
            system_prompt = str(msg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
            settings = msg.get("settings") or {}
            sampler = normalize_sampler(settings)
            tts_mode = str(settings.get("tts_mode") or "server").strip().lower()
            server_tts_enabled = tts_mode == "server"
            tts_engine = normalize_tts_engine(settings.get("tts_engine"))
            audio_mode = str(msg.get("audio_mode") or settings.get("audio_input_mode") or "").strip().lower()
            text_raw = str(msg.get("text") or "").strip()
            transcript_raw = str(msg.get("transcription") or msg.get("transcript") or "").strip()
            if audio_mode == "native":
                user_text = text_raw
            else:
                user_text = (text_raw or transcript_raw).strip()
            if user_text in {"…", "...", "with audio"}:
                user_text = ""

            # Do not let phantom microphone/VAD events become model turns.
            # If there is no text, no raw audio, and no visual attachment, ignore the event.
            if not user_text and not msg.get("audio") and not extract_image_infos(msg, limit=1):
                continue

            llama_session = get_llama_session(chat_id, system_prompt)

            request_id = str(msg.get("request_id") or f"r-{int(time.time() * 1000)}")
            audio_log(
                "rx_user_turn",
                request_id=request_id,
                chat_id=chat_id,
                msg_type=str(msg.get("type") or ""),
                audio_mode=audio_mode or "stt",
                has_audio=bool(msg.get("audio")),
                audio_b64_len=len(str(msg.get("audio") or "")),
                audio_seconds=msg.get("audio_seconds"),
                text_len=len(text_raw),
                transcript_len=len(transcript_raw),
                user_text_len=len(user_text),
                tts_engine=tts_engine,
            )
            t0 = time.time()
            llm_queue: asyncio.Queue[str | None] = asyncio.Queue()

            def stream_worker():
                try:
                    assert llama_session is not None
                    ensure_llama_model(msg.get("llama_model_path") or msg.get("model_path"), msg.get("llama_mmproj_path") or msg.get("mmproj_path"))
                    messages = build_llama_messages(llama_session, system_prompt, msg, user_text)
                    if TEXT_STREAMING and LLAMA_STREAMING:
                        for piece in llama_chat_stream(messages, sampler):
                            if piece:
                                loop.call_soon_threadsafe(llm_queue.put_nowait, piece)
                    else:
                        text = llama_chat_once(messages, sampler)
                        if text:
                            loop.call_soon_threadsafe(llm_queue.put_nowait, text)
                except Exception as exc:
                    print(f"LLM backend error: {exc}")
                    loop.call_soon_threadsafe(llm_queue.put_nowait, f"\n[LLM error: {exc}]\n")
                finally:
                    loop.call_soon_threadsafe(llm_queue.put_nowait, None)

            threading.Thread(target=stream_worker, daemon=True).start()

            audio_started = False
            sentence_index = 0
            tts_total_time = 0.0
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
            seen_tts_sentences: set[str] = set()

            async def tts_worker():
                nonlocal audio_started, sentence_index, tts_total_time
                request_backend = None
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None or request_cancelled(request_id):
                        break
                    clean_sentence = sanitize_tts_text(sentence)
                    key = re.sub(r"\W+", "", clean_sentence.lower())[:240]
                    if len(clean_sentence) < 2 or key in seen_tts_sentences:
                        continue
                    seen_tts_sentences.add(key)

                    if request_backend is None:
                        # TTS loading MUST NOT block the LLM/text stream, but this worker is
                        # separate from the LLM stream. So it may wait/load in executor and
                        # still the user sees text_delta immediately. Do NOT skip voice forever.
                        request_backend = get_cached_tts_backend(tts_engine, settings)
                        if request_backend is None:
                            start_tts_background_load(tts_engine, settings)
                            try:
                                request_backend = await loop.run_in_executor(None, lambda: get_tts_backend(tts_engine, settings))
                            except Exception as exc:
                                err = f"TTS backend error ({tts_engine}): {exc}"
                                print(err)
                                with suppress(Exception):
                                    await ws.send_text(json.dumps({"type": "error", "request_id": request_id, "message": err}, ensure_ascii=False))
                                break

                    if not audio_started:
                        await ws.send_text(json.dumps({
                            "type": "audio_start",
                            "request_id": request_id,
                            "sample_rate": request_backend.sample_rate,
                        }))
                        audio_started = True
                    tts0 = time.time()
                    pcm = await loop.run_in_executor(None, lambda s=clean_sentence, b=request_backend: b.generate(s))
                    tts_total_time += time.time() - tts0
                    if request_cancelled(request_id):
                        break
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_text(json.dumps({
                        "type": "audio_chunk",
                        "request_id": request_id,
                        "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                        "index": sentence_index,
                    }))
                    sentence_index += 1

            tts_task = asyncio.create_task(tts_worker())
            raw_full = ""
            clean_sent_text = ""
            sentence_buffer = ""
            seen_text_sentences: set[str] = set()

            if not TEXT_STREAMING:
                while True:
                    if request_cancelled(request_id):
                        break
                    piece = await llm_queue.get()
                    if piece is None:
                        break
                    raw_full += piece

                llm_time = time.time() - t0
                final_clean = clean_generated_response(raw_full)

                if not request_cancelled(request_id):
                    if llama_session is not None:
                        append_llama_history(llama_session, msg, user_text, final_clean)
                    await ws.send_text(json.dumps({
                        "type": "text_final",
                        "request_id": request_id,
                        "text": final_clean,
                        "llm_time": round(llm_time, 2),
                        "tts_time": 0.0,
                        "sampler": sampler,
                        "backend": LLM_BACKEND,
                    }, ensure_ascii=False))
                    if server_tts_enabled:
                        for sentence in final_tts_sentences(final_clean):
                            await tts_queue.put(sentence)

                await tts_queue.put(None)
                await tts_task

                if server_tts_enabled and not request_cancelled(request_id):
                    await ws.send_text(json.dumps({
                        "type": "audio_end",
                        "request_id": request_id,
                        "tts_time": round(tts_total_time, 2),
                    }))
                continue

            # Streaming path: show token deltas immediately and feed TTS with low-latency chunks.
            # llama.cpp may stream either true deltas or cumulative prefixes; normalize once here.
            visible_text = ""
            sentence_buffer = ""
            seen_text_sentences: set[str] = set()

            async def enqueue_tts_chunk(chunk: str) -> None:
                clean_sentence = strip_thinking_and_controls(chunk, final=True).strip()
                if not clean_sentence:
                    return
                key = re.sub(r"\W+", "", clean_sentence.lower())[:240]
                if key in seen_text_sentences:
                    return
                seen_text_sentences.add(key)
                if TTS_STREAMING and server_tts_enabled:
                    await tts_queue.put(clean_sentence)

            while True:
                if request_cancelled(request_id):
                    break
                piece = await llm_queue.get()
                if piece is None:
                    break

                piece = strip_thinking_and_controls(str(piece), final=False)
                if not piece:
                    continue
                delta, visible_text = normalize_stream_delta(piece, visible_text)
                if not delta:
                    continue

                await ws.send_text(json.dumps({
                    "type": "text_delta",
                    "request_id": request_id,
                    "text": delta,
                }, ensure_ascii=False))

                sentence_buffer += delta
                chunks, sentence_buffer = extract_speak_chunks(sentence_buffer, force=False, first=(len(seen_text_sentences) == 0))
                for chunk in chunks:
                    await enqueue_tts_chunk(chunk)

            llm_time = time.time() - t0
            final_clean = clean_generated_response(visible_text)

            # Flush any remaining spoken tail. This is what makes voice start before full answer,
            # but still completes the final fragment after generation ends.
            tail_chunks, sentence_buffer = extract_speak_chunks(sentence_buffer, force=True, first=(len(seen_text_sentences) == 0))
            for chunk in tail_chunks:
                await enqueue_tts_chunk(chunk)

            if not request_cancelled(request_id):
                if llama_session is not None:
                    append_llama_history(llama_session, msg, user_text, final_clean)
                # Send final text immediately after LLM ends; do not wait for TTS to finish.
                await ws.send_text(json.dumps({
                    "type": "text_final",
                    "request_id": request_id,
                    "text": final_clean,
                    "llm_time": round(llm_time, 2),
                    "tts_time": round(tts_total_time, 2),
                    "sampler": sampler,
                    "backend": LLM_BACKEND,
                }, ensure_ascii=False))

            await tts_queue.put(None)
            await tts_task

            if server_tts_enabled and not request_cancelled(request_id):
                await ws.send_text(json.dumps({"type": "audio_end", "request_id": request_id, "tts_time": round(tts_total_time, 2)}))

    except Exception as exc:
        print(f"WebSocket session error: {exc}")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        ws=os.environ.get("UVICORN_WS_IMPL", "websockets"),
        ws_ping_interval=float(os.environ.get("UVICORN_WS_PING_INTERVAL", "20")),
        ws_ping_timeout=float(os.environ.get("UVICORN_WS_PING_TIMEOUT", "20")),
        ws_max_size=int(os.environ.get("UVICORN_WS_MAX_SIZE", str(32 * 1024 * 1024))),
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        access_log=os.environ.get("UVICORN_ACCESS_LOG", "1").strip().lower() not in {"0", "false", "no", "off"},
    )
