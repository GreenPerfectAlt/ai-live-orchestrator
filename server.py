"""
server.py — self-contained FastAPI + WebSocket orchestrator.
No external project modules required (only optional tts.py / tts_silero.py).
Works with BOTH frontend versions (old parlor.jarvis UI and new ailo UI).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import http.client
import io
import json
import os
import re
import socket
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    import tts as _supertonic_mod
    TTS_SUPERTONIC_ERR = None
except Exception as exc:  # pragma: no cover
    _supertonic_mod = None
    TTS_SUPERTONIC_ERR = exc

try:
    import tts_silero as _silero_mod
    TTS_SILERO_ERR = None
except Exception as exc:  # pragma: no cover
    _silero_mod = None
    TTS_SILERO_ERR = exc


# ── env helpers (clean keys, no trailing spaces) ─────────────────────────
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
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# ── config ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
LLM_BACKEND = "llama_cpp"
MODEL_PATH = env_str("MODEL_PATH", str(PROJECT_ROOT / "models" / "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"))
MODEL_LABEL = env_str("MODEL_LABEL", Path(MODEL_PATH).name)
LAUNCHER_NAME = env_str("LAUNCHER_NAME", "unknown.bat")

LLAMA_HOST = env_str("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = env_int("LLAMA_PORT", 8080)
LLAMA_BASE_URL = env_str("LLAMA_BASE_URL", f"http://{LLAMA_HOST}:{LLAMA_PORT}/v1").rstrip("/")
LLAMA_MODEL = env_str("LLAMA_MODEL", env_str("LLAMA_MODEL_ID", "local-gemma"))
LLAMA_API_KEY = env_str("LLAMA_API_KEY", "no-key")
LLAMA_CTX_SIZE = env_int("LLAMA_CTX_SIZE", 4096)
LLAMA_HISTORY_TURNS = env_int("LLAMA_HISTORY_TURNS", 8)
LLAMA_REQUEST_TIMEOUT = env_float("LLAMA_REQUEST_TIMEOUT", 600)
LLAMA_STARTUP_TIMEOUT = env_float("LLAMA_STARTUP_TIMEOUT", 240)
LLAMA_ENABLE_AUDIO = env_bool("LLAMA_ENABLE_AUDIO", True)
LLAMA_ENABLE_IMAGES = env_bool("LLAMA_ENABLE_IMAGES", True)
LLAMA_MAX_IMAGES = env_int("LLAMA_MAX_IMAGES", 8)
LLAMA_REASONING_FORMAT = env_str("LLAMA_REASONING_FORMAT", "none").strip() or "none"
LLAMA_STREAMING = env_bool("LLAMA_STREAMING", True)
TEXT_STREAMING = env_bool("TEXT_STREAMING", True)
AUDIO_DEBUG = env_bool("PARLOR_AUDIO_DEBUG", True)

# Only Gemma 4 (E2B/E4B/12B) understands input_audio in llama.cpp.
def _model_supports_native_audio(model_path: str) -> bool:
    name = Path(model_path).name.lower()
    return "gemma" in name and any(x in name for x in ("e2b", "e4b", "12b"))

LLAMA_SUPPORTS_AUDIO = _model_supports_native_audio(MODEL_PATH) and LLAMA_ENABLE_AUDIO

# LIVE-FIX: bat hardcodes TTS_EARLY_CHARS=30 / TTS_SPLIT_ON_COMMA=0 → pause
# after first words. Clamp here; bat stays untouched.
TTS_STREAMING = env_bool("TTS_STREAMING", True)
TTS_EARLY_CHARS = min(env_int("TTS_EARLY_CHARS", 10), 12)
TTS_LONG_CHARS = min(env_int("TTS_LONG_CHARS", 60), 70)
TTS_MAX_CHARS = min(env_int("TTS_MAX_CHARS", 160), 180)
TTS_SPLIT_ON_COMMA = True
TTS_SENTENCE_STREAMING = env_bool("TTS_SENTENCE_STREAMING", True)
TTS_ENGINE_DEFAULT = env_str("TTS_ENGINE", "silero")

STT_ENGINE = env_str("STT_ENGINE", "faster_whisper").strip().lower()
STT_MODEL = env_str("STT_MODEL", "small").strip() or "small"
STT_LANG = env_str("STT_LANG", "ru").strip() or None
STT_COMPUTE_TYPE = env_str("STT_COMPUTE_TYPE", "int8").strip() or "int8"
STT_BEAM_SIZE = env_int("STT_BEAM_SIZE", 3)
STT_VAD_ENABLE = env_bool("STT_VAD_ENABLE", True)
STT_VAD_SILENCE_MS = env_int("STT_VAD_SILENCE_MS", 450)
STT_DEVICE = env_str("STT_DEVICE", "cpu").strip() or "cpu"
STT_THREADS = env_int("STT_THREADS", 2)

TALKING_HEAD_ENABLED = env_bool("TALKING_HEAD_ENABLED", False)
TALKING_HEAD_WS = env_str("TALKING_HEAD_WS", "ws://127.0.0.1:8001")

DEFAULT_MAX_OUTPUT_TOKENS = env_int("LLM_MAX_OUTPUT_TOKENS", 0)
DEFAULT_REPEAT_PENALTY = env_float("LLM_REPEAT_PENALTY", 1.18)
DEFAULT_REPEAT_LAST_N = env_int("LLM_REPEAT_LAST_N", 192)
DEFAULT_SAMPLER = {
    "temperature": env_float("LLM_TEMPERATURE", 1.0),
    "top_p": env_float("LLM_TOP_P", 1.0),
    "top_k": env_int("LLM_TOP_K", 0),
    "min_p": env_float("LLM_MIN_P", 0.08),
    "typical_p": env_float("LLM_TYPICAL_P", 1.0),
    "seed": env_int("LLM_SEED", 0),
}
DEFAULT_SYSTEM_PROMPT = (
    "Ты — голосовой ИИ-ассистент. Отвечай естественно, напрямую и по текущему сообщению пользователя. "
    "Всегда учитывай предыдущие сообщения чата. "
    "Если вопрос простой — отвечай коротко; если пользователь просит объяснить, перечислить или продолжить — отвечай полно. "
    "Обычно говори по-русски, но если пользователь пишет или говорит на другом языке — отвечай на нём. "
    "Не показывай скрытые рассуждения, thought/think/reasoning-каналы, XML/служебные теги."
)

# ── garbage-cleanup regexes ───────────────────────────────────────────────
CONTROL_TOKEN_RE = re.compile(r"<\|/?[^>\n]{0,80}?\|>", re.IGNORECASE)
XML_CONTROL_RE = re.compile(r"</?(?:tool|tool_call|tool_response|turn|channel|assistant|model|user|system)[^>]*>", re.IGNORECASE)
THINK_PAIR_RE = re.compile(r"<(think|thought|analysis|reasoning)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_RE = re.compile(r"<(think|thought|analysis|reasoning)\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
CHANNEL_PAIR_RE = re.compile(r"<\|channel>\s*(?:thought|analysis|reasoning)\b.*?<channel\|>", re.IGNORECASE | re.DOTALL)
CHANNEL_OPEN_RE = re.compile(r"<\|channel>\s*(?:thought|analysis|reasoning)\b.*$", re.IGNORECASE | re.DOTALL)
LABEL_RE = re.compile(r"\b(?:Транскрипция|Ответ|Assistant|Model)\s*:\s*", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t]{2,}")
SENTENCE_END_RE = re.compile(r"(?<=[.!?…])(?:\s+|$)|\n+")


def audio_log(event: str, **kwargs) -> None:
    if not AUDIO_DEBUG:
        return
    try:
        print("[VOICE] " + json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)
    except Exception as exc:
        print(f"[VOICE] log failed: {exc}", flush=True)


# ── llama-server HTTP helpers ─────────────────────────────────────────────
def _host_port() -> tuple[str, int]:
    try:
        no_scheme = LLAMA_BASE_URL.split("//")[-1].split("/")[0]
        host, _, port = no_scheme.partition(":")
        return host, int(port or 80)
    except (ValueError, IndexError):
        return LLAMA_HOST, LLAMA_PORT


def _chat_blocking(messages: list, max_tokens: int = 1, sampler: dict | None = None) -> dict:
    sampler = sampler or {}
    body = {
        "model": LLAMA_MODEL, "messages": messages, "stream": False,
        "cache_prompt": True, "max_tokens": max_tokens,
        "temperature": sampler.get("temperature", 1.0),
    }
    host, port = _host_port()
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    data = json.loads(conn.getresponse().read() or b"{}")
    conn.close()
    return data


class ChatStream:
    """Streaming chat with REAL cancel(): socket shutdown aborts generation."""

    def __init__(self, body: dict):
        self.body = body
        self.conn: http.client.HTTPConnection | None = None
        self.cancelled = False
        self.prompt_tokens: int | None = None

    def run(self, on_delta, on_reasoning=None) -> None:
        host, port = _host_port()
        self.conn = http.client.HTTPConnection(host, port, timeout=LLAMA_REQUEST_TIMEOUT)
        self.conn.request("POST", "/v1/chat/completions", json.dumps(self.body),
                          {"Content-Type": "application/json", "Authorization": f"Bearer {LLAMA_API_KEY}"})
        resp = self.conn.getresponse()
        if resp.status != 200:
            body = resp.read()[:300]
            self.conn.close()
            raise RuntimeError(f"llama-server HTTP {resp.status}: {body!r}")
        try:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens"):
                    self.prompt_tokens = usage["prompt_tokens"]
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    if isinstance(delta, dict):
                        text = delta.get("content")
                        reason = delta.get("reasoning_content") or delta.get("reasoning")
                    else:
                        text, reason = None, None
                    if reason and on_reasoning:
                        on_reasoning(reason)
                    if text:
                        on_delta(text)
        except Exception as e:
            if not self.cancelled:
                print(f"LLM stream ended early: {type(e).__name__}: {e}")
        finally:
            try:
                self.conn.close()
            except OSError:
                pass

    def cancel(self) -> None:
        self.cancelled = True
        try:
            if self.conn and self.conn.sock:
                self.conn.sock.shutdown(socket.SHUT_RDWR)
            if self.conn:
                self.conn.close()
        except OSError:
                pass


def wait_for_llama_server() -> None:
    deadline = time.time() + LLAMA_STARTUP_TIMEOUT
    last: Exception | None = None
    host, port = _host_port()
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/v1/models")
            conn.getresponse().read()
            conn.close()
            print(f"✅ llama.cpp server ready: {LLAMA_BASE_URL}")
            return
        except Exception as exc:
            last = exc
            time.sleep(1.0)
    raise RuntimeError(f"llama.cpp server not reachable at {LLAMA_BASE_URL}. Last error: {last}")


# ── STT (faster-whisper, multi-model, offline cache) ─────────────────────
_stt_models: dict[str, Any] = {}
_stt_failed: set[str] = set()
_stt_lock = threading.Lock()


def _stt_get(model_name: str | None = None):
    if STT_ENGINE != "faster_whisper":
        return None
    name = (model_name or STT_MODEL or "small").strip().lower() or "small"
    with _stt_lock:
        if name in _stt_models:
            return _stt_models[name]
        if name in _stt_failed:
            return None
    try:
        from faster_whisper import WhisperModel
        local = Path(name).expanduser()
        use_path = str(local) if local.is_dir() else name
        print(f"🎙 Loading server STT: faster-whisper model={name}, device={STT_DEVICE}, compute={STT_COMPUTE_TYPE}")
        model = WhisperModel(use_path, device=STT_DEVICE, compute_type=STT_COMPUTE_TYPE, cpu_threads=max(1, STT_THREADS))
        print(f"✅ Server STT ready: faster-whisper ({name})")
        with _stt_lock:
            _stt_models[name] = model
        return model
    except Exception as exc:
        print(f"⚠️ faster-whisper model '{name}' unavailable: {exc}")
        with _stt_lock:
            _stt_failed.add(name)
        return None


def stt_preload() -> None:
    threading.Thread(target=_stt_get, daemon=True).start()


def transcribe_audio(audio_f32, model_name: str | None = None) -> str:
    model = _stt_get(model_name)
    if model is None or audio_f32 is None or audio_f32.size < 3200:
        return ""
    try:
        with _stt_lock:
            segments, _info = model.transcribe(
                audio_f32, language=STT_LANG, beam_size=max(1, STT_BEAM_SIZE),
                vad_filter=STT_VAD_ENABLE,
                vad_parameters={"min_silence_duration_ms": STT_VAD_SILENCE_MS},
                condition_on_previous_text=False,
            )
            return " ".join(seg.text for seg in segments).strip()
    except Exception as exc:
        audio_log("whisper_transcribe_failed", err=str(exc))
        return ""


# ── WAV utils ─────────────────────────────────────────────────────────────
def wav_to_float32(b64: str) -> np.ndarray:
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def valid_audio(b64: str | None) -> bool:
    if not b64:
        return False
    return len(b64) * 3 // 4 > 44 + 3200


def pad_tail_silence(b64: str, seconds: float = 0.3) -> str:
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    silence = b"\x00" * (params.sampwidth * params.nchannels * int(seconds * params.framerate))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(frames + silence)
    return base64.b64encode(out.getvalue()).decode()


# ── text cleaning / TTS chunking ─────────────────────────────────────────
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
    return SPACE_RE.sub(" ", text).strip()


def clean_generated_response(text: str) -> str:
    text = strip_thinking_and_controls(text, final=True)
    text = re.sub(r"([.!?…])\s*\1+", r"\1", text)
    text = re.sub(r"\b([\wА-Яа-яЁё-]{2,})(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    return SPACE_RE.sub(" ", text).strip()


def normalize_stream_delta(chunk_text: str, emitted_text: str) -> tuple[str, str]:
    text = chunk_text or ""
    if not text:
        return "", emitted_text
    if text.startswith(emitted_text):
        return text[len(emitted_text):], text
    if emitted_text.endswith(text):
        return "", emitted_text
    max_overlap = min(len(emitted_text), len(text), 512)
    for n in range(max_overlap, 0, -1):
        if emitted_text.endswith(text[:n]):
            return text[n:], emitted_text + text[n:]
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
                rest = []
                if carry:
                    rest.append(carry)
                if tail.strip():
                    rest.append(tail.strip())
                return out, " ".join(rest).strip()
    first_chars = max(6, TTS_EARLY_CHARS)
    target_chars = max(first_chars + 20, TTS_LONG_CHARS)
    max_chars = max(target_chars + 40, TTS_MAX_CHARS)
    out: list[str] = []
    threshold = first_chars if first else target_chars
    min_sentence = 8 if first else 20
    while len(buf) >= threshold:
        window_len = min(len(buf), max_chars)
        window = buf[:window_len]
        split_at = -1
        ends = [m.end() for m in SENTENCE_END_RE.finditer(window) if m.end() >= min_sentence]
        if ends:
            split_at = ends[0]
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


# ── TTS backends cache ───────────────────────────────────────────────────
_tts_backends: dict[str, Any] = {}
_tts_lock = threading.Lock()
_tts_loading: set[str] = set()


def normalize_tts_engine(value: Any) -> str:
    raw = str(value or TTS_ENGINE_DEFAULT).strip().lower()
    return "silero" if raw in {"silero", "silero_ru", "silero-ru", "ru"} else "supertonic"


def _silero_speaker(value: Any) -> str:
    if _silero_mod is not None:
        return _silero_mod._normalize_speaker(value)
    raw = str(value or "xenia").strip().lower()
    mapping = {"f4": "baya", "f3": "xenia", "female": "xenia", "male": "aidar"}
    raw = mapping.get(raw, raw)
    return raw if raw in {"baya", "xenia", "kseniya", "aidar", "eugene"} else "xenia"


def tts_cache_key(engine: str, settings: dict) -> str:
    engine = normalize_tts_engine(engine)
    if engine == "silero":
        speaker = _silero_speaker(settings.get("silero_speaker") or settings.get("voice"))
        speed = max(0.85, min(1.2, float(settings.get("silero_speed") or 1.0)))
        sr = int(settings.get("silero_sample_rate") or 24000)
        model_id = str(settings.get("silero_model") or "v5_5_ru").strip() or "v5_5_ru"
        return f"silero:{model_id}:{speaker}:{sr}:{speed:.3f}"
    return "supertonic"


def get_tts_backend(engine: str, settings: dict | None = None):
    settings = settings or {}
    engine = normalize_tts_engine(engine)
    key = tts_cache_key(engine, settings)
    with _tts_lock:
        backend = _tts_backends.get(key)
        if backend is not None:
            return backend
    if engine == "silero":
        if _silero_mod is None:
            raise RuntimeError(f"Silero backend unavailable: {TTS_SILERO_ERR}")
        backend = _silero_mod.load(
            model_id=str(settings.get("silero_model") or "v5_5_ru"),
            speaker=settings.get("silero_speaker") or settings.get("voice"),
            sample_rate=int(settings.get("silero_sample_rate") or 24000),
            speed=float(settings.get("silero_speed") or 1.0),
        )
    else:
        if _supertonic_mod is None:
            raise RuntimeError(f"Supertonic backend unavailable: {TTS_SUPERTONIC_ERR}")
        backend = _supertonic_mod.load()
    with _tts_lock:
        _tts_backends[key] = backend
    return backend


def start_tts_background_load(engine: str, settings: dict) -> None:
    key = tts_cache_key(engine, settings)
    with _tts_lock:
        if key in _tts_backends or key in _tts_loading:
            return
        _tts_loading.add(key)

    def _load() -> None:
        try:
            print(f"🔊 Background TTS load started: {key}")
            get_tts_backend(engine, settings)
            print(f"✅ Background TTS ready: {key}")
        except Exception as exc:
            print(f"⚠️ Background TTS load failed ({key}): {exc}")
        finally:
            with _tts_lock:
                _tts_loading.discard(key)

    threading.Thread(target=_load, daemon=True).start()


# ── sampler / sessions / messages ────────────────────────────────────────
def _clamp_f(v, d, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return d

def _clamp_i(v, d, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return d


def normalize_sampler(settings: dict | None) -> dict:
    settings = settings or {}
    d = DEFAULT_SAMPLER
    max_out = _clamp_i(settings.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS), DEFAULT_MAX_OUTPUT_TOKENS, -1, 32768)
    if max_out > 0:
        max_out = max(32, min(32768, max_out))
    return {
        "temperature": _clamp_f(settings.get("temperature"), d["temperature"], 0.0, 2.0),
        "top_p": _clamp_f(settings.get("top_p"), d["top_p"], 0.0, 1.0),
        "top_k": _clamp_i(settings.get("top_k"), d["top_k"], 0, 256),
        "min_p": _clamp_f(settings.get("min_p"), d["min_p"], 0.0, 1.0),
        "typical_p": _clamp_f(settings.get("typical_p"), d["typical_p"], 0.0, 1.0),
        "seed": _clamp_i(settings.get("seed"), d["seed"], 0, 2_147_483_647),
        "max_output_tokens": max_out,
        "repeat_penalty": _clamp_f(settings.get("repeat_penalty"), DEFAULT_REPEAT_PENALTY, 1.0, 2.0),
        "repeat_last_n": _clamp_i(settings.get("repeat_last_n"), DEFAULT_REPEAT_LAST_N, 0, 32768),
    }


@dataclass
class LlamaSession:
    chat_id: str
    prompt_id: str
    system_prompt: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)


CONTEXT_HEADROOM = max(512, min(2000, LLAMA_CTX_SIZE // 8))


def stable_prompt_id(prompt: str, sampler: dict) -> str:
    payload = json.dumps({"prompt": prompt, "sampler": sampler}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def rotate_history(history: list) -> list:
    if len(history) <= 3:
        return history
    keep = 1 + max(2, 3 * (len(history) - 1) // 4)
    while keep > 3 and history[-(keep - 1)].get("role") != "user":
        keep -= 1
    return [history[0]] + history[-(keep - 1):]


EXTRA_CONTEXT: str = ""


class ContextPayload(BaseModel):
    text: str


def llama_system_prompt(system_prompt: str) -> str:
    base = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
    if EXTRA_CONTEXT:
        base += f"\n\n[DOCUMENT CONTEXT (RAG)]:\n{EXTRA_CONTEXT}"
    return base


def normalize_client_history(msg: dict) -> list[dict[str, str]]:
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


def extract_image_infos(msg: dict, limit: int | None = None) -> list[dict[str, str]]:
    if limit is None:
        limit = max(1, LLAMA_MAX_IMAGES)
    infos: list[dict[str, str]] = []

    def add_item(item: Any, default_source: str = "image") -> None:
        if len(infos) >= limit:
            return
        source, blob = default_source, None
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


def data_uri(b64: str) -> str:
    b64 = (b64 or "").strip()
    return b64 if b64.startswith("data:") else f"data:image/jpeg;base64,{b64}"


def build_user_content(user_text: str, image_b64: str | None, audio_b64s: list[str]) -> list[dict]:
    parts: list[dict] = []
    if image_b64:
        parts.append({"type": "image_url", "image_url": {"url": data_uri(image_b64)}})
    for b in audio_b64s:
        if valid_audio(b):
            parts.append({"type": "input_audio", "input_audio": {"data": b, "format": "wav"}})
    if user_text:
        prompt_text = user_text.strip()
    elif audio_b64s:
        prompt_text = "Прослушай аудио пользователя и ответь на него."
    elif image_b64:
        prompt_text = "Посмотри на изображение и ответь на запрос."
    else:
        prompt_text = "Продолжи разговор по последней реплике."
    parts.append({"type": "text", "text": prompt_text})
    return parts


def build_llama_messages(session: LlamaSession, msg: dict, user_text: str,
                         image_b64: str | None, audio_b64s: list[str]) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": llama_system_prompt(session.system_prompt)}]
    client_history = normalize_client_history(msg)
    if client_history:
        messages.extend(client_history)
    elif LLAMA_HISTORY_TURNS > 0:
        messages.extend(session.history[-LLAMA_HISTORY_TURNS * 2:])
    messages.append({"role": "user", "content": build_user_content(user_text, image_b64, audio_b64s)})
    return messages


def llama_payload(messages: list, sampler: dict, *, stream: bool) -> dict:
    payload: dict[str, Any] = {
        "model": LLAMA_MODEL, "messages": messages, "stream": stream,
        "cache_prompt": True, "reasoning_format": LLAMA_REASONING_FORMAT,
        "temperature": sampler["temperature"], "top_p": sampler["top_p"],
        "top_k": sampler["top_k"], "min_p": sampler["min_p"],
        "typical_p": sampler["typical_p"],
        "repeat_penalty": sampler["repeat_penalty"], "repeat_last_n": sampler["repeat_last_n"],
    }
    if sampler["seed"]:
        payload["seed"] = sampler["seed"]
    if sampler["max_output_tokens"] > 0:
        payload["max_tokens"] = sampler["max_output_tokens"]
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def estimate_tokens(messages: list) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content) // 4 + 8
            continue
        if isinstance(content, list):
            for p in content:
                t = p.get("type") if isinstance(p, dict) else None
                if t == "text":
                    total += len(p.get("text", "")) // 4
                elif t == "input_audio":
                    total += (len(p.get("input_audio", {}).get("data", "")) * 3 // 4 // 32000) * 32
                else:
                    total += 300
        total += 8
    return total


async def send_to_talking_head(text: str, audio_b64: str, sample_rate: int) -> None:
    try:
        import websockets
        async with websockets.connect(TALKING_HEAD_WS, open_timeout=1.5, close_timeout=1.0) as ws_th:
            await ws_th.send(json.dumps({"action": "speak", "text": text, "audio": audio_b64, "sample_rate": sample_rate}))
    except Exception:
        pass


# ── app ───────────────────────────────────────────────────────────────────
def load_models() -> None:
    print("🧠 LLM backend: llama.cpp only")
    wait_for_llama_server()
    print(f"🧠 Model: {Path(MODEL_PATH).name}")
    print(f"🎙 Native audio: {'✅ supported' if LLAMA_SUPPORTS_AUDIO else '❌ STT pipeline forced'}")
    print("🔊 TTS: Supertonic 3 + Silero RU")
    start_tts_background_load(TTS_ENGINE_DEFAULT, {
        "silero_speaker": env_str("SILERO_SPEAKER", "xenia"),
        "silero_speed": env_str("SILERO_SPEED", "1.0"),
        "silero_sample_rate": env_str("SILERO_SAMPLE_RATE", "24000"),
        "silero_model": env_str("SILERO_MODEL", "v5_5_ru"),
    })
    stt_preload()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(content=(PROJECT_ROOT / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    return {
        "backend": LLM_BACKEND, "model_label": MODEL_LABEL, "model": LLAMA_MODEL,
        "launcher_name": LAUNCHER_NAME, "text_streaming": TEXT_STREAMING,
        "llama_streaming": LLAMA_STREAMING, "tts_streaming": TTS_STREAMING,
        "tts_engine": TTS_ENGINE_DEFAULT, "supports_native_audio": LLAMA_SUPPORTS_AUDIO,
    }


@app.post("/api/upload_context")
async def upload_context(payload: ContextPayload):
    global EXTRA_CONTEXT
    EXTRA_CONTEXT = payload.text[:16000]
    return {"ok": True, "chars": len(EXTRA_CONTEXT)}


@app.post("/api/clear_context")
async def clear_context():
    global EXTRA_CONTEXT
    EXTRA_CONTEXT = ""
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # Push status immediately: fixes "model: loading / bat: unknown" in BOTH frontends.
    await ws.send_text(json.dumps({
        "type": "app_status", "backend": LLM_BACKEND, "model_label": MODEL_LABEL,
        "model": LLAMA_MODEL, "launcher_name": LAUNCHER_NAME,
        "supports_native_audio": LLAMA_SUPPORTS_AUDIO,
    }))
    interrupted = asyncio.Event()
    cancelled_requests: set[str] = set()
    msg_queue: asyncio.Queue[dict | None] = asyncio.Queue()
    sessions: dict[str, LlamaSession] = {}
    active: dict[str, Any] = {"stream": None}
    frame_image: str | None = None
    speech_chunks: list[str] = []
    priming = {"active": False}
    loop = asyncio.get_running_loop()

    async def receiver() -> None:
        nonlocal frame_image, speech_chunks
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                mtype = msg.get("type")
                if mtype == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                elif mtype == "interrupt":
                    rid = str(msg.get("request_id") or "").strip()
                    if rid:
                        cancelled_requests.add(rid)
                    interrupted.set()
                    stream = active.get("stream")
                    if stream:
                        stream.cancel()
                elif mtype == "ready":
                    interrupted.clear()
                elif mtype == "reset":
                    sessions.pop(str(msg.get("chat_id") or "default"), None)
                elif mtype == "frame":
                    if msg.get("image") and LLAMA_ENABLE_IMAGES:
                        frame_image = msg["image"]
                        speech_chunks = []
                        prime(frame_image, [])
                elif mtype == "speech_chunk":
                    if msg.get("seq") == 0:
                        speech_chunks = []
                    if valid_audio(msg.get("audio")):
                        speech_chunks.append(msg["audio"])
                        prime(frame_image, speech_chunks)
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    def prime(image_b64: str | None, audio_b64s: list[str]) -> None:
        if priming["active"] or (not image_b64 and not audio_b64s):
            return
        priming["active"] = True

        def _run() -> None:
            try:
                sess = next(iter(sessions.values()), None)
                system = llama_system_prompt(sess.system_prompt if sess else "")
                msgs = [{"role": "system", "content": system}]
                if sess:
                    msgs += list(sess.history)
                msgs.append({"role": "user", "content": build_user_content("", image_b64, audio_b64s)})
                _chat_blocking(msgs, max_tokens=1)
            except Exception as e:
                print(f"⚡ cache priming failed (non-fatal): {e}")
            finally:
                priming["active"] = False

        loop.create_task(loop.run_in_executor(None, _run))

    def get_session(chat_id: str, system_prompt: str) -> LlamaSession:
        prompt_id = stable_prompt_id(system_prompt, {"backend": "llama_cpp"})
        existing = sessions.get(chat_id)
        if existing and existing.prompt_id == prompt_id:
            return existing
        if len(sessions) >= 64:
            sessions.pop(next(iter(sessions)))
        session = LlamaSession(chat_id=chat_id, prompt_id=prompt_id, system_prompt=system_prompt)
        sessions[chat_id] = session
        return session

    def request_cancelled(request_id: str) -> bool:
        return bool(request_id and request_id in cancelled_requests) or interrupted.is_set()

    def decode_concat(b64s: list[str]):
        parts = []
        for b in b64s:
            try:
                parts.append(wav_to_float32(b))
            except Exception:
                pass
        return np.concatenate(parts) if parts else None

    recv_task = asyncio.create_task(receiver())
    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break
            interrupted.clear()

            # ── TTS replay (🔊): speak stored text WITHOUT the LLM ──
            if msg.get("type") == "tts":
                tts_rid = str(msg.get("request_id") or f"tts-{int(time.time() * 1000)}")
                tts_text = strip_thinking_and_controls(str(msg.get("text") or ""), final=True).strip()
                tts_settings = {
                    "silero_speaker": msg.get("silero_speaker") or msg.get("voice"),
                    "silero_speed": msg.get("silero_speed"), "voice": msg.get("voice"),
                }
                engine_name = normalize_tts_engine(msg.get("tts_engine"))
                if tts_text:
                    try:
                        backend = await loop.run_in_executor(None, lambda: get_tts_backend(engine_name, tts_settings))
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

            # ── normal turn ──
            chat_id = str(msg.get("chat_id") or "default")[:80]
            system_prompt = str(msg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
            settings = msg.get("settings") or {}
            sampler = normalize_sampler(settings)
            server_tts_enabled = str(settings.get("tts_mode") or "server").strip().lower() == "server"
            tts_engine = normalize_tts_engine(settings.get("tts_engine"))
            stt_model_choice = str(settings.get("stt_model") or STT_MODEL).strip() or STT_MODEL
            user_text = str(msg.get("text") or "").strip()

            audio_b64s = list(speech_chunks)
            speech_chunks = []
            if valid_audio(msg.get("audio")):
                audio_b64s.append(msg["audio"])
            image_b64 = None
            if LLAMA_ENABLE_IMAGES:
                extracted = extract_image_infos(msg, limit=1)
                image_b64 = extracted[0]["blob"] if extracted else (msg.get("image") or frame_image)
            frame_image = None

            if not user_text and not audio_b64s and not image_b64:
                continue

            session = get_session(chat_id, system_prompt)
            request_id = str(msg.get("request_id") or f"r-{int(time.time() * 1000)}")
            t0 = time.time()

            # Audio pipeline: native / stt / hybrid; force STT for non-Gemma models.
            audio_pipeline = str(settings.get("audio_pipeline") or msg.get("audio_mode") or "native").strip().lower()
            if audio_pipeline not in {"native", "stt", "hybrid"}:
                audio_pipeline = "native"
            if not LLAMA_SUPPORTS_AUDIO and audio_pipeline in {"native", "hybrid"}:
                audio_pipeline = "stt"
            use_native_audio = LLAMA_SUPPORTS_AUDIO and audio_pipeline in {"native", "hybrid"}
            need_text_for_llm = bool(audio_b64s) and not use_native_audio

            if audio_b64s:
                try:
                    audio_b64s[-1] = pad_tail_silence(audio_b64s[-1])
                except Exception:
                    pass

            llm_audio = audio_b64s if use_native_audio else []
            transcript = ""
            if audio_b64s and need_text_for_llm:
                audio_arr = await loop.run_in_executor(None, decode_concat, audio_b64s)
                transcript = await loop.run_in_executor(None, transcribe_audio, audio_arr, stt_model_choice)
                await ws.send_text(json.dumps({
                    "type": "transcription", "request_id": request_id,
                    "text": transcript, "error": ("" if transcript else "stt_empty"),
                }, ensure_ascii=False))
                if transcript:
                    user_text = user_text or transcript
                elif not LLAMA_SUPPORTS_AUDIO:
                    audio_b64s = []
                    llm_audio = []

            messages = build_llama_messages(session, msg, user_text, image_b64, llm_audio)
            est = estimate_tokens(messages)
            if est > LLAMA_CTX_SIZE - 2 * CONTEXT_HEADROOM and session.history:
                rotated = rotate_history(session.history)
                if len(rotated) < len(session.history):
                    print(f"Context near limit (est {est}) — dropping {len(session.history) - len(rotated)} oldest messages")
                    session.history = rotated
                    messages = build_llama_messages(session, msg, user_text, image_b64, llm_audio)

            llm_queue: asyncio.Queue[Any] = asyncio.Queue()

            def stream_worker() -> None:
                body = llama_payload(messages, sampler, stream=True)
                stream = ChatStream(body)
                active["stream"] = stream
                try:
                    def on_delta(text: str) -> None:
                        loop.call_soon_threadsafe(llm_queue.put_nowait, text)
                    def on_reasoning(text: str) -> None:
                        loop.call_soon_threadsafe(llm_queue.put_nowait, ("__think__", text))
                    stream.run(on_delta, on_reasoning)
                except Exception as exc:
                    loop.call_soon_threadsafe(llm_queue.put_nowait, f"\n[LLM error: {exc}]\n")
                finally:
                    active["stream"] = None
                    loop.call_soon_threadsafe(llm_queue.put_nowait, ("__usage__", stream.prompt_tokens))
                    loop.call_soon_threadsafe(llm_queue.put_nowait, None)

            threading.Thread(target=stream_worker, daemon=True).start()

            if audio_b64s and not need_text_for_llm:
                async def bg_transcribe() -> None:
                    try:
                        audio_arr = await loop.run_in_executor(None, decode_concat, audio_b64s)
                        txt = await loop.run_in_executor(None, stt.transcribe, audio_arr, stt_model_choice)
                        if not request_cancelled(request_id):
                            await ws.send_text(json.dumps({
                                "type": "transcription", "request_id": request_id,
                                "text": txt, "error": ("" if txt else "stt_empty"),
                            }, ensure_ascii=False))
                    except Exception as exc:
                        audio_log("whisper_worker_failed", err=str(exc))
                asyncio.create_task(bg_transcribe())

            audio_started = False
            sentence_index = 0
            tts_total_time = 0.0
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
            seen_tts: set[str] = set()

            async def tts_worker() -> None:
                nonlocal audio_started, sentence_index, tts_total_time
                request_backend = None
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None or request_cancelled(request_id):
                        break
                    clean_sentence = sanitize_tts_text(sentence)
                    key = re.sub(r"\W+", "", clean_sentence.lower())[:240]
                    if len(clean_sentence) < 2 or key in seen_tts:
                        continue
                    seen_tts.add(key)
                    if request_backend is None:
                        try:
                            request_backend = await loop.run_in_executor(None, lambda: get_tts_backend(tts_engine, settings))
                        except Exception:
                            break
                    if not audio_started and request_backend:
                        await ws.send_text(json.dumps({"type": "audio_start", "request_id": request_id, "sample_rate": request_backend.sample_rate}))
                        audio_started = True
                    tts0 = time.time()
                    pcm = await loop.run_in_executor(None, lambda s=clean_sentence, b=request_backend: b.generate(s))
                    tts_total_time += time.time() - tts0
                    if request_cancelled(request_id):
                        break
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    audio_b64 = base64.b64encode(pcm_int16.tobytes()).decode()
                    await ws.send_text(json.dumps({"type": "audio_chunk", "request_id": request_id, "audio": audio_b64, "index": sentence_index}))
                    sentence_index += 1
                    if TALKING_HEAD_ENABLED and request_backend:
                        await send_to_talking_head(clean_sentence, audio_b64, request_backend.sample_rate)

            tts_task = asyncio.create_task(tts_worker())
            visible_text = ""
            sentence_buffer = ""
            seen_text: set[str] = set()

            async def enqueue_tts_chunk(chunk: str) -> None:
                clean_sentence = strip_thinking_and_controls(chunk, final=True).strip()
                if not clean_sentence:
                    return
                key = re.sub(r"\W+", "", clean_sentence.lower())[:240]
                if key in seen_text:
                    return
                seen_text.add(key)
                if TTS_STREAMING and server_tts_enabled:
                    await tts_queue.put(clean_sentence)

            while True:
                if request_cancelled(request_id):
                    break
                piece = await llm_queue.get()
                if piece is None:
                    break
                if isinstance(piece, tuple) and piece and piece[0] == "__usage__":
                    continue
                if isinstance(piece, tuple) and piece and piece[0] == "__think__":
                    await ws.send_text(json.dumps({"type": "thinking_delta", "request_id": request_id, "text": piece[1]}, ensure_ascii=False))
                    continue
                piece = strip_thinking_and_controls(str(piece), final=False)
                if not piece:
                    continue
                delta, visible_text = normalize_stream_delta(piece, visible_text)
                if not delta:
                    continue
                await ws.send_text(json.dumps({"type": "text_delta", "request_id": request_id, "text": delta}, ensure_ascii=False))
                sentence_buffer += delta
                chunks, sentence_buffer = extract_speak_chunks(sentence_buffer, force=False, first=(len(seen_text) == 0))
                for chunk in chunks:
                    await enqueue_tts_chunk(chunk)

            llm_time = time.time() - t0
            final_clean = clean_generated_response(visible_text)
            tail_chunks, sentence_buffer = extract_speak_chunks(sentence_buffer, force=True, first=(len(seen_text) == 0))
            for chunk in tail_chunks:
                await enqueue_tts_chunk(chunk)

            if not request_cancelled(request_id):
                if final_clean.strip():
                    session.history.append({"role": "user", "content": (user_text.strip() or "[voice/media]")[:2000]})
                    session.history.append({"role": "assistant", "content": final_clean[:4000]})
                await ws.send_text(json.dumps({
                    "type": "text_final", "request_id": request_id, "text": final_clean,
                    "llm_time": round(llm_time, 2), "tts_time": round(tts_total_time, 2),
                    "sampler": sampler, "backend": LLM_BACKEND, "transcription": transcript or None,
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
    uvicorn.run(app, host="127.0.0.1", port=8000,
                ws=os.environ.get("UVICORN_WS_IMPL", "websockets"), log_level="info")