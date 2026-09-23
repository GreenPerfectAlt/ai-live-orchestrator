"""Supertonic 3 TTS backend for Parlor."""

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

def _project_root() -> Path:
    return Path(__file__).resolve().parent

def _supertonic_model_dir() -> Path:
    env_dir = os.environ.get("SUPERTONIC_CACHE_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (_project_root() / "models" / "supertonic3").resolve()


class TTSBackend:
    sample_rate: int = 44100

    def generate(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class SupertonicBackend(TTSBackend):
    def __init__(self, lang: str = "auto", voice: str = "F4", speed: float = 1.0, total_steps: int = 3):
        model_dir = _supertonic_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SUPERTONIC_CACHE_DIR", str(model_dir))

        try:
            from supertonic import TTS
        except Exception as exc:
            raise RuntimeError("Не установлен Supertonic. Выполни: python -m pip install -U supertonic") from exc

        self._lang = lang
        self._voice_name = voice
        self._speed = speed
        self._total_steps = total_steps
        self._lock = threading.Lock()
        self._model_dir = model_dir

        try:
            self._tts: Any = TTS(model="supertonic-3", model_dir=str(model_dir), auto_download=False)
        except Exception as exc:
            raise RuntimeError(f"Supertonic 3 model files not ready in: {model_dir}") from exc

        self._voice_style = self._tts.get_voice_style(voice_name=voice)

        # Warmup
        _wav, _duration = self._tts.synthesize(
            text="Привет.",
            voice_style=self._voice_style,
            lang=_select_lang_for_text("Привет.", lang),
            total_steps=total_steps,
            speed=speed,
            max_chunk_length=300,
            verbose=False,
        )
        print(f"✅ TTS: Supertonic 3, lang={lang}, voice={voice}, steps={total_steps}, {self.sample_rate} Hz")

    @staticmethod
    def _as_float32_mono(wav: Any) -> np.ndarray:
        arr = np.asarray(wav, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return np.clip(arr, -1.0, 1.0)

    def generate(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
    ) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        voice_name = voice or self._voice_name
        voice_style = self._tts.get_voice_style(voice_name=voice_name) if voice_name != self._voice_name else self._voice_style

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
    lang = os.environ.get("TTS_LANG", "auto")
    voice = os.environ.get("TTS_VOICE", "F4")
    speed = float(os.environ.get("TTS_SPEED", "1.0"))
    steps = int(os.environ.get("TTS_STEPS", "3"))
    return SupertonicBackend(lang=lang, voice=voice, speed=speed, total_steps=steps)