"""
tts.py — Supertonic 3 TTS backend (multilingual diffusion TTS, ONNX).

Two live-mode fixes vs the old version:
- sample_rate is READ FROM THE MODEL instead of the hardcoded 44100.
  A wrong rate makes the frontend play audio at the wrong speed/pitch
  (the "hell sound" complaint).
- total_steps is clamped to [4, 10]: the .bat hardcodes TTS_STEPS=1,
  which leaves the diffusion pass half-finished = robotic artifacts.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

# LIVE-FIX: below 4 diffusion steps the output is artifact-heavy.
MIN_TOTAL_STEPS = 4
MAX_TOTAL_STEPS = 10

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _select_lang_for_text(text: str, configured_lang: str) -> str:
    cfg = (configured_lang or "auto").strip().lower()
    if cfg not in {"auto", "mixed", "na"}:
        return cfg
    cyr = len(_CYRILLIC_RE.findall(text or ""))
    lat = len(_LATIN_RE.findall(text or ""))
    if cyr > 0 and lat > 0:
        return "na"
    if lat > 0 and cyr == 0:
        return "en"
    if cyr > 0:
        return "ru"
    return "na"


def _supertonic_model_dir() -> Path:
    env_dir = os.environ.get("SUPERTONIC_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (Path(__file__).resolve().parent / "models" / "supertonic3").resolve()


class TTSBackend:
    """Unified TTS interface: generate() -> float32 mono PCM."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str | None = None,
                 speed: float | None = None, lang: str | None = None) -> np.ndarray:
        raise NotImplementedError


class SupertonicBackend(TTSBackend):
    def __init__(self, lang: str = "auto", voice: str = "F4",
                 speed: float = 1.0, total_steps: int = 5):
        model_dir = _supertonic_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SUPERTONIC_CACHE_DIR", str(model_dir))

        try:
            from supertonic import TTS
        except Exception as exc:
            raise RuntimeError(
                "Supertonic is not installed. Run: python -m pip install -U supertonic"
            ) from exc

        self._lang = lang
        self._voice_name = voice
        self._speed = max(0.8, min(1.2, float(speed or 1.0)))
        # LIVE-FIX: clamp the diffusion step count (bat hardcodes 1).
        self._total_steps = max(MIN_TOTAL_STEPS, min(MAX_TOTAL_STEPS, int(total_steps or 5)))
        self._lock = threading.Lock()
        self._model_dir = model_dir

        try:
            self._tts: Any = TTS(model="supertonic-3", model_dir=str(model_dir), auto_download=False)
        except Exception as exc:
            raise RuntimeError(f"Supertonic 3 model files not ready in: {model_dir}") from exc

        # LIVE-FIX: real vocoder rate, not a hardcoded constant.
        self.sample_rate = int(getattr(self._tts, "sample_rate", 24000))

        self._voice_style = self._tts.get_voice_style(voice_name=voice)

        # Warmup: first synthesize pays pipeline init; do it here so the
        # first real sentence is hot.
        try:
            self._tts.synthesize(
                text="Привет.",
                voice_style=self._voice_style,
                lang=_select_lang_for_text("Привет.", lang),
                total_steps=self._total_steps,
                speed=self._speed,
                max_chunk_length=300,
                verbose=False,
            )
        except Exception as exc:
            print(f"[TTS] Supertonic warmup warning: {exc}")

        print(
            f"✅ TTS: Supertonic 3, lang={lang}, voice={voice}, "
            f"steps={self._total_steps}, {self.sample_rate} Hz"
        )

    @staticmethod
    def _as_float32_mono(wav: Any) -> np.ndarray:
        arr = np.asarray(wav, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        
        # Если значения выходят за пределы [-1.0, 1.0], значит это int16
        # Делим на 32768.0, чтобы нормализовать звук, иначе clip создаст адский шум
        if arr.size > 0 and np.max(np.abs(arr)) > 1.0:
            arr = arr / 32768.0
            
        return np.clip(arr, -1.0, 1.0)

    def generate(self, text: str, voice: str | None = None,
                 speed: float | None = None, lang: str | None = None) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        voice_name = voice or self._voice_name
        if voice_name != self._voice_name:
            voice_style = self._tts.get_voice_style(voice_name=voice_name)
        else:
            voice_style = self._voice_style
        # Serialized: a concurrent synth would steal CPU from the LLM.
        with self._lock:
            wav, _duration = self._tts.synthesize(
                text=text,
                voice_style=voice_style,
                lang=lang or _select_lang_for_text(text, self._lang),
                total_steps=self._total_steps,
                speed=speed or self._speed,
                max_chunk_length=300,
                verbose=False,
            )
        return self._as_float32_mono(wav)


def load() -> TTSBackend:
    """Entry point used by server.py's TTS backend cache."""
    lang = os.environ.get("TTS_LANG", "auto")
    voice = os.environ.get("TTS_VOICE", "F4")
    speed = float(os.environ.get("TTS_SPEED", "1.0"))
    steps = int(os.environ.get("TTS_STEPS", "5"))
    return SupertonicBackend(lang=lang, voice=voice, speed=speed, total_steps=steps)