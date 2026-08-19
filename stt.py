"""
stt.py — offline speech-to-text via faster-whisper.

Design goals (ported from the project's original server.py, hardened
with parlor v2's reliability rules):
- multi-model cache: any of tiny/base/small/turbo/medium/large-v3 can be
  selected per-request from the frontend; each model loads once and stays
  resident.
- fully offline after first download: the .bat points HF_HOME at
  models\\.hf_cache, so huggingface_hub resolves cached blobs without
  network. A local CTranslate2 directory path is accepted as-is.
- a failed model load is remembered and never retried mid-turn (a broken
  STT must degrade to "no chat text", never to a 500 or a hang).
- transcription is serialized under one lock: ctranslate2 sessions are
  not thread-safe, and two concurrent decodes would thrash the CPU cores
  the LLM is using.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    STT_BEAM_SIZE,
    STT_COMPUTE_TYPE,
    STT_DEVICE,
    STT_ENGINE,
    STT_LANG,
    STT_MODEL,
    STT_THREADS,
    STT_VAD_ENABLE,
    env_int,
)

# Silence duration (ms) the whisper VAD requires before cutting a segment.
# The .bat hardcodes STT_VAD_SILENCE_MS=450; reading it here keeps the
# server-side segmentation aligned with the frontend's VAD.
STT_VAD_SILENCE_MS = env_int("STT_VAD_SILENCE_MS", 450)

# Minimum audio length (~100ms at 16kHz) below which transcription is
# skipped — shorter clips are breaths/clicks and only waste CPU.
MIN_SAMPLES = 3200

_models: dict[str, Any] = {}
_failed: set[str] = set()
_lock = threading.Lock()


def available() -> bool:
    """Whether the configured STT engine is faster-whisper at all."""
    return STT_ENGINE == "faster_whisper"


def _resolve_name(model_name: str | None) -> str:
    return (model_name or STT_MODEL or "small").strip().lower() or "small"


def get_model(model_name: str | None = None) -> Any | None:
    """Lazy-load and cache a WhisperModel per name.

    Accepts either a hub id (tiny/base/small/turbo/medium/large-v3),
    resolved through the HF cache (offline after first download), or a
    local directory containing a CTranslate2 conversion (always offline).
    Returns None (and remembers the failure) instead of raising.
    """
    name = _resolve_name(model_name)
    if not available():
        return None
    with _lock:
        if name in _models:
            return _models[name]
        if name in _failed:
            return None
    try:
        from faster_whisper import WhisperModel

        # A real directory = local CTranslate2 model: never touch network.
        local = Path(name).expanduser()
        use_path = str(local) if local.is_dir() else name

        print(
            f"🎙 Loading server STT: faster-whisper model={name}, "
            f"device={STT_DEVICE}, compute={STT_COMPUTE_TYPE}"
        )
        model = WhisperModel(
            use_path,
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE,
            cpu_threads=max(1, STT_THREADS),
        )
        print(f"✅ Server STT ready: faster-whisper ({name})")
        with _lock:
            _models[name] = model
        return model
    except Exception as exc:
        # Remember the failure: a missing/corrupt model must not be
        # retried on every voice turn (each retry costs seconds).
        print(f"⚠️ faster-whisper model '{name}' unavailable: {exc}")
        with _lock:
            _failed.add(name)
        return None


def preload(model_name: str | None = None) -> None:
    """Background warm-up so the first voice turn doesn't pay model load.
    Safe to call at startup; failures are swallowed (get_model remembers)."""
    threading.Thread(target=get_model, args=(model_name,), daemon=True).start()


def transcribe(audio_f32: np.ndarray | None, model_name: str | None = None) -> str:
    """Transcribe float32 mono audio (16kHz) to text. Returns '' on any
    failure path — the caller treats '' as "no chat text", and the LLM
    still receives the raw audio in native/hybrid pipelines."""
    if not available() or audio_f32 is None or audio_f32.size < MIN_SAMPLES:
        return ""
    model = get_model(model_name)
    if model is None:
        return ""
    try:
        # Serialized: ctranslate2 is not thread-safe, and a concurrent
        # decode would steal CPU from the LLM mid-turn.
        with _lock:
            segments, _info = model.transcribe(
                audio_f32,
                language=STT_LANG,
                beam_size=max(1, STT_BEAM_SIZE),
                vad_filter=STT_VAD_ENABLE,
                vad_parameters={"min_silence_duration_ms": STT_VAD_SILENCE_MS},
                # Never let the previous turn's words leak into this one —
                # with a live mic that's exactly how echo loops start.
                condition_on_previous_text=False,
            )
            return " ".join(seg.text for seg in segments).strip()
    except Exception as exc:
        print(f"[STT] transcription failed: {exc}")
        return ""


def transcribe_wav_b64(b64: str | None, model_name: str | None = None) -> str:
    """Convenience wrapper: decode a base64 WAV (via pipeline's wave-module
    parser) and transcribe it. Imported lazily to avoid an import cycle."""
    if not b64:
        return ""
    from pipeline import wav_to_float32

    try:
        audio = wav_to_float32(b64)
    except Exception as exc:
        # Malformed WAV: report and bail — a bad clip must never poison
        # the turn (parlor v2's valid_audio rule).
        print(f"[STT] wav decode failed: {exc}")
        return ""
    return transcribe(audio, model_name)