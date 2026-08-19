"""
tts_silero.py — Silero RU TTS backend (torch, CPU).

Offline-first load order:
1. local torch.package file (models\silero\v5_5_ru.pt — the .bat points
   SILERO_CACHE_DIR there) — zero network;
2. torch.hub (snakers4/silero-models) when SILERO_USE_HUB=1 and the hub
   cache is warm;
3. direct URL download as the last resort (first run only).

The backend exposes the same interface as tts.py (generate() -> float32
mono PCM at self.sample_rate), so server.py treats both interchangeably.
"""
from __future__ import annotations

import os
import random as _random
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

SPACE_RE = re.compile(r"\s+")
# Markdown / control glyphs Silero would otherwise try to pronounce.
CONTROL_RE = re.compile(r"[#*_`<>|{}\[\]\'\"]+")

# Silero v5 RU ships exactly these voices.
ALLOWED_SPEAKERS = {"aidar", "baya", "kseniya", "xenia", "eugene"}


def _bool_env(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cache_dir() -> Path:
    return Path(
        os.environ.get(
            "SILERO_CACHE_DIR",
            str(Path(__file__).resolve().parent / "models" / "silero"),
        )
    ).expanduser()


def _clean_text(text: str) -> str:
    """Normalize text for the vocoder: strip markup, collapse spaces,
    guarantee a terminal punctuation mark (Silero prosody needs it)."""
    text = str(text or " ").strip()
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("—", "-")
    text = SPACE_RE.sub(" ", text).strip()
    if text and text[-1] not in ".!?…,:;":
        text += "."
    return text


def _as_float32_mono(audio: Any) -> np.ndarray:
    """Coerce any torch/numpy output shape to 1-D float32 in [-1, 1]."""
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.05:
        # Some checkpoints emit int-ish ranges; renormalize instead of
        # hard-clipping (clipping would add audible buzz).
        arr = arr / peak
    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)


def _speed_resample(audio: np.ndarray, speed: float) -> np.ndarray:
    """Linear-interpolation speed change. Skipped entirely at 1.0 — no
    pointless resampling of every chunk on the hot path."""
    try:
        speed = float(speed)
    except Exception:
        speed = 1.0
    if not np.isfinite(speed) or abs(speed - 1.0) < 0.01:
        return audio
    speed = max(0.85, min(1.2, speed))
    if audio.size < 4:
        return audio
    new_len = max(2, int(round(audio.size / speed)))
    x_old = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _normalize_speaker(speaker: str) -> str:
    """Map legacy/alias names to real Silero v5 voices; 'random' picks one
    per backend instance (apply_tts itself has no 'random' speaker)."""
    raw = str(speaker or " ").strip().lower()
    mapping = {"f4": "baya", "f3": "xenia", "female": "xenia", "male": "aidar"}
    raw = mapping.get(raw, raw)
    if raw == "random":
        return _random.choice(sorted(ALLOWED_SPEAKERS))
    return raw if raw in ALLOWED_SPEAKERS else "xenia"


class SileroBackend:
    def __init__(
        self,
        model_id: str = "v5_5_ru",
        speaker: str = "xenia",
        sample_rate: int = 24000,
        speed: float = 1.0,
    ) -> None:
        self.model_id = model_id or "v5_5_ru"
        self.speaker = _normalize_speaker(speaker)
        # Silero v5 supports exactly these output rates.
        self.sample_rate = int(sample_rate or 24000)
        if self.sample_rate not in {8000, 24000, 48000}:
            self.sample_rate = 24000
        self.speed = float(speed or 1.0)
        self.put_accent = _bool_env("SILERO_PUT_ACCENT", True)
        self.put_yo = _bool_env("SILERO_PUT_YO", True)
        self._lock = threading.Lock()

        self._torch = self._import_torch()
        self._model = self._load_model()
        try:
            self._model.to("cpu")
            self._model.eval()
        except Exception:
            pass

        # Warmup: first synthesize pays graph init; do it here so the
        # first real sentence is hot.
        try:
            self.generate("Привет.")
        except Exception as exc:
            print(f"[TTS] Silero warmup warning: {exc}")

        print(
            f"✅ TTS: Silero RU, model={self.model_id}, speaker={self.speaker}, "
            f"sample_rate={self.sample_rate}, speed={self.speed}"
        )

    def _import_torch(self):
        import torch

        # Cap torch threads: the LLM owns the cores; TTS must not starve it.
        threads = int(os.environ.get("TTS_THREADS", os.environ.get("LLAMA_THREADS", "4")))
        try:
            torch.set_num_threads(max(1, min(threads, 8)))
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        # The .bat points TORCH_HOME at models\.torch — keep hub downloads
        # inside the project cache.
        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            try:
                torch.hub.set_dir(torch_home)
            except Exception:
                pass
        try:
            # torch.hub fork-guard false-positives under some launchers.
            torch.hub._validate_not_a_forked_repo = lambda *a, **k: True
        except Exception:
            pass
        return torch

    def _try_direct_package(self, model_path: Path):
        """Load a local torch.package checkpoint — fully offline path."""
        torch = self._torch
        importer = torch.package.PackageImporter(str(model_path))
        return importer.load_pickle("tts_models", "model")

    def _try_torch_hub(self):
        torch = self._torch
        print("[TTS] Loading Silero via torch.hub...")
        kwargs = dict(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=self.model_id,
        )
        try:
            result = torch.hub.load(**kwargs, trust_repo=True)
        except TypeError:
            result = torch.hub.load(**kwargs)
        if isinstance(result, (list, tuple)):
            return result[0]
        return result

    def _load_model(self):
        torch = self._torch
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        model_path = Path(
            os.environ.get("SILERO_MODEL_PATH", str(cache / f"{self.model_id}.pt"))
        ).expanduser()
        model_url = os.environ.get(
            "SILERO_MODEL_URL",
            f"https://models.silero.ai/models/tts/ru/{self.model_id}.pt",
        )

        # 1) Local package — the .bat's SILERO_CACHE_DIR lands here.
        if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
            try:
                print(f"[TTS] Loading Silero local package: {model_path}")
                return self._try_direct_package(model_path)
            except Exception as exc:
                print(f"[TTS] Local Silero package failed: {exc}")

        # 2) torch.hub with a warm cache (SILERO_USE_HUB=1 in the .bat).
        if _bool_env("SILERO_USE_HUB", True):
            try:
                return self._try_torch_hub()
            except Exception as exc:
                print(f"[TTS] torch.hub Silero load failed: {exc}")

        # 3) First run only: download the .pt, then load it directly.
        if not model_path.exists() or model_path.stat().st_size <= 1024 * 1024:
            print(f"[TTS] Downloading Silero model directly: {model_url}")
            torch.hub.download_url_to_file(model_url, str(model_path), progress=True)
        return self._try_direct_package(model_path)

    def generate(self, text: str) -> np.ndarray:
        text = _clean_text(text)
        if not text:
            return np.zeros(1, dtype=np.float32)
        # Serialized: the jit model is not thread-safe, and a concurrent
        # synth would steal CPU from the LLM mid-turn.
        with self._lock:
            torch = self._torch
            with torch.inference_mode():
                try:
                    audio = self._model.apply_tts(
                        text=text,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate,
                        put_accent=self.put_accent,
                        put_yo=self.put_yo,
                    )
                except TypeError:
                    # Older checkpoint signature without accent flags.
                    audio = self._model.apply_tts(
                        text=text,
                        speaker=self.speaker,
                        sample_rate=self.sample_rate,
                    )
        arr = _as_float32_mono(audio)
        arr = _speed_resample(arr, self.speed)
        return arr


def load(
    model_id: str | None = None,
    speaker: str | None = None,
    sample_rate: int | None = None,
    speed: float | None = None,
) -> SileroBackend:
    """Entry point used by server.py's TTS backend cache."""
    model_id = model_id or os.environ.get("SILERO_MODEL", "v5_5_ru")
    speaker = speaker or os.environ.get("SILERO_SPEAKER", "xenia")
    sample_rate = int(sample_rate or os.environ.get("SILERO_SAMPLE_RATE", "24000"))
    speed = float(speed if speed is not None else os.environ.get("SILERO_SPEED", "1.0"))
    return SileroBackend(
        model_id=model_id, speaker=speaker, sample_rate=sample_rate, speed=speed
    )