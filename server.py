"""
Ai-Live-Orchestrator — llama.cpp-only local voice/text/vision chat.
The browser can attach camera/screen/PDF/video frames to each text or voice turn.
This build talks only to llama-server /v1/chat/completions.
Thinking/reasoning отключён полностью; strip_thinking_and_controls страхует от
случайных thought-тегов в выводе модели.
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
DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "0"))
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
TTS_EARLY_CHARS = int(os.environ.get("TTS_EARLY_CHARS", "15"))
TTS_LONG_CHARS = int(os.environ.get("TTS_LONG_CHARS", "80"))
TTS_MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "180"))
TTS_SPLIT_ON_COMMA = os.environ.get("TTS_SPLIT_ON_COMMA", "1").strip().lower() in {"1", "true", "yes", "on"}
TTS_SENTENCE_STREAMING = os.environ.get("TTS_SENTENCE_STREAMING", "1").strip().lower() in {"1", "true", "yes", "on"}
LLAMA_ENABLE_IMAGES = os.environ.get("LLAMA_ENABLE_IMAGES", "1").strip().lower() not in {"0", "false", "no", "off"}
LLAMA_MAX_IMAGES = int(os.environ.get("LLAMA_MAX_IMAGES", "8"))
LLAMA_STARTUP_TIMEOUT = float(os.environ.get("LLAMA_STARTUP_TIMEOUT", "240"))
LLAMA_REQUEST_TIMEOUT = float(os.environ.get("LLAMA_REQUEST_TIMEOUT", "600"))
LLAMA_HISTORY_TURNS = int(os.environ.get("LLAMA_HISTORY_TURNS", "8"))
LLAMA_REASONING_FORMAT = os.environ.get("LLAMA_REASONING_FORMAT", "none").strip() or "none"

DEFAULT_SAMPLER = {
    "temperature": float(os.environ.get("LLM_TEMPERATURE", "1.0")),
    "top_p": float(os.environ.get("LLM_TOP_P", "1.0")),
    "top_k": int(os.environ.get("LLM_TOP_K", "0")),
    "min_p": float(os.environ.get("LLM_MIN_P", "0.08")),
    "typical_p": float(os.environ.get("LLM_TYPICAL_P", "1.0")),
    "seed": int(os.environ.get("LLM_SEED", "0")),
}

DEFAULT_SYSTEM_PROMPT = (
    "Ты — голосовой ИИ-ассистент Parlor. Отвечай естественно, напрямую и по текущему сообщению пользователя. "
    "Всегда учитывай предыдущие сообщения чата. "
    "Если вопрос простой — отвечай коротко; если пользователь просит объяснить, перечислить или продолжить — отвечай полно. "
    "Обычно говори по-русски, но если пользователь пишет или говорит по-английски — отвечай по-английски. "
    "Не показывай скрытые рассуждения, thought/think/reasoning-каналы, XML/служебные теги."
)

# Regex'ы нужны ТОЛЬКО как страховка: вырезают thought/channel теги, если модель
# их всё же сгенерит. Это НЕ фича «думания», а очистка мусора из ответа.
CONTROL_TOKEN_RE = re.compile(r"<\|/?[^>\n]{0,80}?\|>", re.IGNORECASE)
XML_CONTROL_RE = re.compile(r"</?(?:tool|tool_call|tool_response|turn|channel|assistant|model|user|system)[^>]*>", re.IGNORECASE)
THINK_PAIR_RE = re.compile(r"<(think|thought|analysis|reasoning)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE = re.compile(r"<(think|thought|analysis|reasoning)\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
CHANNEL_PAIR_RE = re.compile(r"<\|channel>\s*(?:thought|analysis|reasoning)\b.*?<channel\|>", re.IGNORECASE | re.DOTALL)
CHANNEL_OPEN_RE = re.compile(r"<\|channel>\s*(?:thought|analysis|reasoning)\b.*$", re.IGNORECASE | re.DOTALL)
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
    raw = str(value or os.environ.get("SILERO_SPEAKER", "xenia")).strip().lower()
    mapping = {"f4": "baya", "f3": "xenia", "female": "xenia", "male": "aidar"}
    raw = mapping.get(raw, raw)
    allowed = {"baya", "xenia", "kseniya", "aidar", "eugene", "random"}
    return raw if raw in allowed else "xenia"


def tts_cache_key(engine: str = "supertonic", settings: dict[str, Any] | None = None) -> str:
    settings = settings or {}
    engine = normalize_tts_engine(engine)
    if engine == "silero":
        speaker = normalize_silero_speaker(settings.get("silero_speaker") or settings.get("voice"))
        speed = clamp_float(settings.get("silero_speed"), float(os.environ.get("SILERO_SPEED", os.environ.get("TTS_SPEED", "1.0"))), 0.85, 1.2)
        sample_rate = clamp_int(settings.get("silero_sample_rate"), int(os.environ.get("SILERO_SAMPLE_RATE", "24000")), 8000, 48000)
        model_id = str(settings.get("silero_model") or os.environ.get("SILERO_MODEL", "v5_5_ru")).strip() or "v5_5_ru"
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
        if key in tts_backends or key in tts_loading_keys:
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
    global tts_backend
    settings = settings or {}
    engine = normalize_tts_engine(engine)
    if engine == "silero":
        if tts_silero is None:
            raise RuntimeError(f"Silero backend is unavailable: {TTS_SILERO_IMPORT_ERROR}")
        speaker = normalize_silero_speaker(settings.get("silero_speaker") or settings.get("voice"))
        speed = clamp_float(settings.get("silero_speed"), float(os.environ.get("SILERO_SPEED", os.environ.get("TTS_SPEED", "1.0"))), 0.85, 1.2)
        sample_rate = clamp_int(settings.get("silero_sample_rate"), int(os.environ.get("SILERO_SAMPLE_RATE", "24000")), 8000, 48000)
        model_id = str(settings.get("silero_model") or os.environ.get("SILERO_MODEL", "v5_5_ru")).strip() or "v5_5_ru"
        key = f"silero:{model_id}:{speaker}:{sample_rate}:{speed:.3f}"
        with tts_backend_lock:
            backend = tts_backends.get(key)
            if backend is None:
                print(f"🔊 Loading TTS backend: Silero RU | speaker={speaker}, speed={speed}, sr={sample_rate}")
                backend = tts_silero.load(model_id=model_id, speaker=speaker, sample_rate=sample_rate, speed=speed)
                tts_backends[key] = backend
            return backend
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


STT_ENGINE = os.environ.get("STT_ENGINE", "faster_whisper").strip().lower()
STT_MODEL = os.environ.get("STT_MODEL", "small").strip() or "small"
STT_LANG = os.environ.get("STT_LANG", "ru").strip() or None
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8").strip() or "int8"
STT_BEAM_SIZE = int(os.environ.get("STT_BEAM_SIZE", "3"))
STT_VAD_ENABLE = os.environ.get("STT_VAD_ENABLE", "1").strip().lower() in {"1", "true", "yes", "on"}
STT_DEVICE = os.environ.get("STT_DEVICE", "cpu").strip() or "cpu"
STT_THREADS = int(os.environ.get("STT_THREADS", "2"))

_whisper_model = None
_whisper_lock = threading.Lock()
_whisper_tried = False


def _get_whisper():
    global _whisper_model, _whisper_tried
    if STT_ENGINE != "faster_whisper":
        return None
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is not None:
            return _whisper_model
        if _whisper_tried:
            return None
        _whisper_tried = True
        try:
            from faster_whisper import WhisperModel
            print(f"🎙 Loading server STT: faster-whisper model={STT_MODEL}, device={STT_DEVICE}, compute={STT_COMPUTE_TYPE}")
            _whisper_model = WhisperModel(STT_MODEL, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE, cpu_threads=max(1, STT_THREADS))
            print("✅ Server STT ready: faster-whisper")
            return _whisper_model
        except Exception as exc:
            print(f"⚠️ faster-whisper unavailable (native mode will work without chat text): {exc}")
            return None


def decode_wav_b64(b64: str):
    try:
        raw = base64.b64decode(b64)
        idx = raw.find(b"data")
        if idx < 0 or idx + 8 > len(raw):
            return None
        size = int.from_bytes(raw[idx + 4:idx + 8], "little")
        pcm = raw[idx + 8:idx + 8 + size]
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return audio if audio.size else None
    except Exception as exc:
        audio_log("wav_decode_failed", err=str(exc))
        return None


def transcribe_audio(audio_f32) -> str:
    model = _get_whisper()
    if model is None or audio_f32 is None or audio_f32.size < 3200:
        return ""
    try:
        with _whisper_lock:
            segments, _info = model.transcribe(
                audio_f32, language=STT_LANG, beam_size=max(1, STT_BEAM_SIZE),
                vad_filter=STT_VAD_ENABLE, condition_on_previous_text=False,
            )
            text = " ".join(seg.text for seg in segments).strip()
        return text
    except Exception as exc:
        audio_log("whisper_transcribe_failed", err=str(exc))
        return ""


def normalize_sampler(settings: dict[str, Any] | None) -> dict[str, Any]:
    settings = settings or {}
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
    text = SPACE_RE.sub(" ", text)
    return text.strip() if final else text


def sanitize_tts_text(text: str) -> str:
    text = strip_thinking_and_controls(text or "", final=True)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+", "", text)
    text = text.replace("*", "").replace("_", "").replace("`", "")
    text = re.sub(r"\s+([.!?…])", r"\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def collapse_generated_repeats(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"([.!?…])\s*\1+", r"\1", text)
    text = re.sub(r"\b([A-Za-zА-Яа-яЁё]{3,})(?:\1\b)+", r"\1", text)
    text = re.sub(r"\b([\wА-Яа-яЁё-]{2,})(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def clean_generated_response(text: str) -> str:
    return collapse_generated_repeats(strip_thinking_and_controls(text, final=True))


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
    buf = SPACE_RE.sub(" ", (buffer or "").strip())
    if not buf:
        return [], ""
    if TTS_SENTENCE_STREAMING:
        complete, tail = extract_sentences(buf)
        if complete:
            min_sentence = 6 if first else 14
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
    first_chars = max(6, int(TTS_EARLY_CHARS))
    target_chars = max(first_chars + 20, int(TTS_LONG_CHARS))
    max_chars = max(target_chars + 40, int(TTS_MAX_CHARS))
    out: list[str] = []
    threshold = first_chars if first else target_chars
    min_sentence = 8 if first else 20
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
        min_sentence = 20
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
            token = "e2b" if "e2b" in model.name.lower() else "e4b" if "e4b" in model.name.lower() else ""
            if token:
                paired = next((mm for mm in mmprojs if token in mm.name.lower()), None)
        rel_model = str(model.relative_to(base)) if model.is_relative_to(base) else str(model)
        rel_mm = str(paired.relative_to(base)) if paired and paired.is_relative_to(base) else (str(paired) if paired else "")
        items.append({
            "id": hashlib.sha1(str(model).encode("utf-8", "ignore")).hexdigest()[:12],
            "name": model.name, "label": model.stem, "path": rel_model,
            "absolute_path": str(model), "mmproj_name": paired.name if paired else "",
            "mmproj_path": rel_mm, "mmproj_absolute_path": str(paired) if paired else "",
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
        raise RuntimeError(f"No GGUF models found in {MODELS_DIR}.")
    chosen = next((m for m in models if "e2b" in m["name"].lower()), models[0])
    return str(chosen.get("path") or ""), str(chosen.get("mmproj_path") or "")


def ensure_llama_model(model_path_value: Any = None, mmproj_path_value: Any = None) -> dict[str, Any]:
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
    signature = hashlib.sha1(json.dumps({
        "model": str(model_path), "mmproj": str(mmproj_path or ""), "host": LLAMA_HOST,
        "port": LLAMA_PORT, "ctx": LLAMA_CTX_SIZE, "threads": LLAMA_THREADS,
        "batch": LLAMA_BATCH_SIZE, "ngl": LLAMA_N_GPU_LAYERS, "extra": LLAMA_EXTRA_ARGS,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    with llama_process_lock:
        if llama_active_signature == signature and server_url_is_ready(timeout=2.0):
            return {"backend": "llama_cpp", "managed": True, "base_url": LLAMA_BASE_URL,
                    "model": model_path.name, "mmproj": mmproj_path.name if mmproj_path else ""}
        stop_managed_llama_server()
        exe = LLAMA_SERVER_EXE.strip().strip('"') or "llama-server.exe"
        exe_path = Path(exe).expanduser()
        exe_cmd = str(exe_path) if exe_path.exists() else (shutil.which(exe) or exe)
        cmd = [exe_cmd, "-m", str(model_path), "--host", LLAMA_HOST, "--port", str(LLAMA_PORT),
               "--ctx-size", str(LLAMA_CTX_SIZE), "--threads", str(LLAMA_THREADS),
               "--batch-size", str(LLAMA_BATCH_SIZE)]
        if LLAMA_N_GPU_LAYERS:
            cmd += ["-ngl", LLAMA_N_GPU_LAYERS]
        if mmproj_path:
            cmd += ["--mmproj", str(mmproj_path)]
        if LLAMA_EXTRA_ARGS:
            cmd += shlex.split(LLAMA_EXTRA_ARGS)
        print("🚀 launching llama-server:")
        print("   " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        try:
            llama_server_process = subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
        except FileNotFoundError as exc:
            raise RuntimeError(f"llama-server.exe not found: {exe_cmd}") from exc
        LLAMA_MODEL = model_path.stem
        wait_for_llama_server()
        llama_active_signature = signature
        llama_active_model_path = str(model_path)
        return {"backend": "llama_cpp", "managed": True, "base_url": LLAMA_BASE_URL,
                "model": model_path.name, "mmproj": mmproj_path.name if mmproj_path else ""}


def load_models() -> None:
    global tts_backend
    print("🧠 LLM backend: llama.cpp only")
    print(f"🚀 llama.cpp: {LLAMA_BASE_URL} | model={LLAMA_MODEL}")
    if not LLAMA_AUTO_START:
        wait_for_llama_server()
    print("🔊 TTS: Supertonic 3 + Silero RU")
    if env_bool("TTS_BACKGROUND_PRELOAD", True):
        start_tts_background_load(os.environ.get("TTS_ENGINE", "silero"), {
            "silero_speaker": os.environ.get("SILERO_SPEAKER", "xenia"),
            "silero_speed": os.environ.get("SILERO_SPEED", "1.0"),
            "silero_sample_rate": os.environ.get("SILERO_SAMPLE_RATE", "24000"),
            "silero_model": os.environ.get("SILERO_MODEL", "v5_5_ru"),
        })


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
        "backend": LLM_BACKEND, "auto_start": LLAMA_AUTO_START, "models_dir": str(MODELS_DIR),
        "base_url": LLAMA_BASE_URL, "active_model_path": llama_active_model_path or "",
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
    if limit is None:
        limit = max(1, LLAMA_MAX_IMAGES)
    infos: list[dict[str, str]] = []

    def add_item(item: Any, default_source: str = "image") -> None:
        if len(infos) >= limit:
            return
        source = default_source
        blob: Any = None
        if isinstance(item, dict):
            blob = item.get("blob") or item.get("image") or item.get("data")
            source = str(item.get("source") or item.get("name") or source).strip().lower()[:40] or source
        else:
            blob = item
        if isinstance(blob, str) and blob.strip():
            infos.append({"source": source, "blob": blob.strip()})

    direct = msg.get("image")
    if isinstance(direct, str) and direct.strip():
        add_item({"source": "image", "blob": direct.strip()})
    for item in msg.get("images") or []:
        add_item(item, "image")
    for item in msg.get("frames") or []:
        add_item(item, "frame")
    return infos[:limit]


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
    if has_audio and LLAMA_ENABLE_AUDIO:
        parts.append({"type": "input_audio", "input_audio": {"data": msg["audio"], "format": "wav"}})
    if user_text:
        prompt_text = user_text.strip()
    elif has_audio:
        prompt_text = "Прослушай аудио пользователя и ответь на него."
    elif has_image:
        prompt_text = "Посмотри на изображение и ответь на запрос."
    else:
        prompt_text = "Продолжи разговор по последней реплике."
    parts.append({"type": "text", "text": prompt_text})
    if has_image and LLAMA_ENABLE_IMAGES:
        for info in image_infos:
            parts.append({"type": "image_url", "image_url": {"url": data_uri_from_base64(info["blob"], "image/jpeg")}})
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
        if not text or text.startswith("[ERROR]"):
            continue
        out.append({"role": role, "content": SPACE_RE.sub(" ", text)[:3000]})
    return out[-max(2, LLAMA_HISTORY_TURNS * 2):]


def llama_system_prompt(system_prompt: str) -> str:
    return system_prompt.strip() or DEFAULT_SYSTEM_PROMPT


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
        "model": LLAMA_MODEL, "messages": messages,
        "temperature": float(sampler["temperature"]), "top_p": float(sampler["top_p"]),
        "top_k": int(sampler["top_k"]), "min_p": float(sampler.get("min_p", 0.08)),
        "typical_p": float(sampler.get("typical_p", 1.0)),
        "repeat_penalty": float(sampler.get("repeat_penalty", DEFAULT_REPEAT_PENALTY)),
        "repeat_last_n": int(sampler.get("repeat_last_n", DEFAULT_REPEAT_LAST_N)),
        "stream": stream, "reasoning_format": LLAMA_REASONING_FORMAT,
    }
    max_out = int(sampler.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    if max_out > 0:
        payload["max_tokens"] = max_out
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
    return str(choice.get("text") or "")


def llama_chat_once(messages: list[dict[str, Any]], sampler: dict[str, Any]) -> str:
    payload = llama_payload(messages, sampler, stream=False)
    obj = http_json("POST", f"{LLAMA_BASE_URL}/chat/completions", payload, timeout=LLAMA_REQUEST_TIMEOUT)
    return extract_llama_text(obj)


def llama_chat_stream(messages: list[dict[str, Any]], sampler: dict[str, Any]) -> Iterator[str]:
    payload = llama_payload(messages, sampler, stream=True)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{LLAMA_BASE_URL}/chat/completions", data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {LLAMA_API_KEY}")
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
            if choices:
                delta = choices[0].get("delta") or {}
                text = delta.get("content", "") if isinstance(delta, dict) else ""
                if text:
                    yield text


def append_llama_history(session: LlamaSession, msg: dict[str, Any], user_text: str, assistant_text: str) -> None:
    if not assistant_text.strip():
        return
    session.history.append({"role": "user", "content": (user_text.strip() or "[Голос/Медиа]")[:2000]})
    session.history.append({"role": "assistant", "content": assistant_text.strip()[:4000]})


@app.get("/api/status")
async def api_status():
    return {
        "backend": LLM_BACKEND, "model_label": MODEL_LABEL, "model": LLAMA_MODEL,
        "launcher_name": LAUNCHER_NAME, "text_streaming": TEXT_STREAMING,
        "llama_streaming": LLAMA_STREAMING, "tts_streaming": TTS_STREAMING,
        "tts_engine": os.environ.get("TTS_ENGINE", "silero"),
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
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
                    await ws.send_text(json.dumps({"type": "pong"}))
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

            # ── TTS-replay (кнопка 🔊 «Listen»): озвучить готовый текст БЕЗ LLM ──
            # Без этой ветки сервер трактует пакет {type:'tts'} как обычный запрос
            # и генерирует НОВЫЙ ответ на текст ответа → «Listen» работает как «continue».
            if msg.get("type") == "tts":
                tts_rid = str(msg.get("request_id") or f"tts-{int(time.time() * 1000)}")
                tts_text = strip_thinking_and_controls(str(msg.get("text") or ""), final=True).strip()
                tts_engine_name = normalize_tts_engine(msg.get("tts_engine"))
                tts_settings = {
                    "silero_speaker": msg.get("silero_speaker") or msg.get("voice"),
                    "silero_speed": msg.get("silero_speed"),
                    "voice": msg.get("voice"),
                }
                if tts_text:
                    try:
                        backend = get_cached_tts_backend(tts_engine_name, tts_settings)
                        if backend is None:
                            backend = await loop.run_in_executor(None, lambda: get_tts_backend(tts_engine_name, tts_settings))
                        sentences = [s for s in extract_speak_chunks(tts_text, force=True)[0] if s.strip()]
                        if sentences:
                            await ws.send_text(json.dumps({"type": "audio_start", "request_id": tts_rid, "sample_rate": backend.sample_rate}))
                            for idx, sentence in enumerate(sentences):
                                pcm = await loop.run_in_executor(None, lambda s=sentence, b=backend: b.generate(s))
                                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                                await ws.send_text(json.dumps({"type": "audio_chunk", "request_id": tts_rid, "audio": base64.b64encode(pcm_int16.tobytes()).decode(), "index": idx}))
                            await ws.send_text(json.dumps({"type": "audio_end", "request_id": tts_rid, "tts_time": 0}))
                    except Exception as exc:
                        print(f"[TTS replay] error: {exc}")
                continue

            chat_id = str(msg.get("chat_id") or "default")[:80]

            chat_id = str(msg.get("chat_id") or "default")[:80]
            system_prompt = str(msg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
            settings = msg.get("settings") or {}
            sampler = normalize_sampler(settings)
            server_tts_enabled = str(settings.get("tts_mode") or "server").strip().lower() == "server"
            tts_engine = normalize_tts_engine(settings.get("tts_engine"))
            text_raw = str(msg.get("text") or "").strip()
            transcript_raw = str(msg.get("transcription") or msg.get("transcript") or "").strip()
            user_text = (text_raw or transcript_raw).strip()

            if not user_text and not msg.get("audio") and not extract_image_infos(msg, limit=1):
                continue

            llama_session = get_llama_session(chat_id, system_prompt)
            request_id = str(msg.get("request_id") or f"r-{int(time.time() * 1000)}")
            t0 = time.time()
            llm_queue: asyncio.Queue[str | None] = asyncio.Queue()

            def stream_worker():
                try:
                    ensure_llama_model(msg.get("llama_model_path") or msg.get("model_path"),
                                       msg.get("llama_mmproj_path") or msg.get("mmproj_path"))
                    messages = build_llama_messages(llama_session, system_prompt, msg, user_text)
                    use_stream = TEXT_STREAMING and LLAMA_STREAMING
                    if use_stream:
                        for piece in llama_chat_stream(messages, sampler):
                            if piece:
                                loop.call_soon_threadsafe(llm_queue.put_nowait, piece)
                    else:
                        text = llama_chat_once(messages, sampler)
                        if text:
                            loop.call_soon_threadsafe(llm_queue.put_nowait, text)
                except Exception as exc:
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
                        request_backend = get_cached_tts_backend(tts_engine, settings)
                        if request_backend is None:
                            start_tts_background_load(tts_engine, settings)
                            try:
                                request_backend = await loop.run_in_executor(None, lambda: get_tts_backend(tts_engine, settings))
                            except Exception:
                                break
                    if not audio_started and request_backend:
                        await ws.send_text(json.dumps({
                            "type": "audio_start", "request_id": request_id,
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
                        "type": "audio_chunk", "request_id": request_id,
                        "audio": base64.b64encode(pcm_int16.tobytes()).decode(), "index": sentence_index,
                    }))
                    sentence_index += 1

            tts_task = asyncio.create_task(tts_worker())

            if (not user_text) and msg.get("audio") and LLAMA_ENABLE_AUDIO:
                async def whisper_worker():
                    try:
                        audio = await loop.run_in_executor(None, lambda: decode_wav_b64(msg["audio"]))
                        txt = await loop.run_in_executor(None, lambda: transcribe_audio(audio))
                        if txt and not request_cancelled(request_id):
                            await ws.send_text(json.dumps({"type": "transcription", "request_id": request_id, "text": txt}, ensure_ascii=False))
                    except Exception as exc:
                        audio_log("whisper_worker_failed", err=str(exc))
                asyncio.create_task(whisper_worker())

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
                # Страховка: вырезаем любые thought/channel теги, если модель их сунула.
                piece = strip_thinking_and_controls(str(piece), final=False)
                if not piece:
                    continue
                delta, visible_text = normalize_stream_delta(piece, visible_text)
                if not delta:
                    continue
                await ws.send_text(json.dumps({
                    "type": "text_delta", "request_id": request_id, "text": delta,
                }, ensure_ascii=False))
                sentence_buffer += delta
                chunks, sentence_buffer = extract_speak_chunks(sentence_buffer, force=False, first=(len(seen_text_sentences) == 0))
                for chunk in chunks:
                    await enqueue_tts_chunk(chunk)

            llm_time = time.time() - t0
            final_clean = clean_generated_response(visible_text)

            tail_chunks, sentence_buffer = extract_speak_chunks(sentence_buffer, force=True, first=(len(seen_text_sentences) == 0))
            for chunk in tail_chunks:
                await enqueue_tts_chunk(chunk)

            if not request_cancelled(request_id):
                append_llama_history(llama_session, msg, user_text, final_clean)
                await ws.send_text(json.dumps({
                    "type": "text_final", "request_id": request_id, "text": final_clean,
                    "llm_time": round(llm_time, 2), "tts_time": round(tts_total_time, 2),
                    "sampler": sampler, "backend": LLM_BACKEND,
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
        app, host="127.0.0.1", port=8000,
        ws=os.environ.get("UVICORN_WS_IMPL", "websockets"), log_level="info",
    )