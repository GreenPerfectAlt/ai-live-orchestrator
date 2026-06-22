"""Silero RU TTS backend for Parlor/Jarvis.

Expected by server.py:
- load(...) -> backend
- backend.sample_rate
- backend.generate(text) -> numpy float32 mono [-1, 1]

This version is intentionally robust:
1) tries local direct .pt package if present;
2) otherwise tries torch.hub Silero;
3) then tries direct download fallback.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[#*_`<>|{}\[\]\\'\"]")


def _bool_env(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cache_dir() -> Path:
    return Path(os.environ.get("SILERO_CACHE_DIR", str(Path(__file__).parent / "models" / "silero"))).expanduser()


def _clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("—", "-")
    text = SPACE_RE.sub(" ", text).strip()
    if text and text[-1] not in ".!?…,:;":
        text += "."
    return text


def _as_float32_mono(audio: Any) -> np.ndarray:
    try:
        import torch  # type: ignore
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
        arr = arr / peak
    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)


def _speed_resample(audio: np.ndarray, speed: float) -> np.ndarray:
    try:
        speed = float(speed)
    except Exception:
        speed = 1.0
    if not np.isfinite(speed) or abs(speed - 1.0) < 0.015:
        return audio
    speed = max(0.85, min(1.2, speed))
    if audio.size < 4:
        return audio
    new_len = max(2, int(round(audio.size / speed)))
    x_old = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _normalize_speaker(speaker: str) -> str:
    raw = str(speaker or "").strip().lower()
    mapping = {
        "f4": "baya",
        "f3": "xenia",
        "female": "baya",
        "woman": "baya",
        "m": "aidar",
        "male": "aidar",
        "man": "aidar",
    }
    raw = mapping.get(raw, raw)
    allowed = {"aidar", "baya", "kseniya", "xenia", "eugene", "random"}
    if raw not in allowed:
        print(f"[TTS] Unknown Silero speaker '{speaker}', using 'baya'.")
        raw = "baya"
    return raw


class SileroBackend:
    def __init__(
        self,
        model_id: str = "v4_ru",
        speaker: str = "baya",
        sample_rate: int = 24000,
        speed: float = 1.0,
    ) -> None:
        self.model_id = model_id or "v4_ru"
        self.speaker = _normalize_speaker(speaker)
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
        except Exception:
            pass
        try:
            self._model.eval()
        except Exception:
            pass

        try:
            self.generate("Привет.")
        except Exception as exc:
            print(f"[TTS] Silero warmup warning: {exc}")

        print(
            f"✅ TTS: Silero RU, model={self.model_id}, speaker={self.speaker}, "
            f"sample_rate={self.sample_rate}, speed={self.speed}"
        )

    def _import_torch(self):
        import torch  # type: ignore

        threads = int(os.environ.get("TTS_THREADS", os.environ.get("LLAMA_THREADS", "4")))
        try:
            torch.set_num_threads(max(1, min(threads, 8)))
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            try:
                torch.hub.set_dir(torch_home)
            except Exception:
                pass

        try:
            torch.hub._validate_not_a_forked_repo = lambda *a, **k: True
        except Exception:
            pass
        return torch

    def _try_direct_package(self, model_path: Path):
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

        model_path = Path(os.environ.get("SILERO_MODEL_PATH", str(cache / f"{self.model_id}.pt"))).expanduser()
        model_url = os.environ.get("SILERO_MODEL_URL", f"https://models.silero.ai/models/tts/ru/{self.model_id}.pt")

        if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
            try:
                print(f"[TTS] Loading Silero local package: {model_path}")
                return self._try_direct_package(model_path)
            except Exception as exc:
                print(f"[TTS] Local Silero package failed: {exc}")

        if _bool_env("SILERO_USE_HUB", True):
            try:
                return self._try_torch_hub()
            except Exception as exc:
                print(f"[TTS] torch.hub Silero load failed: {exc}")

        if not model_path.exists() or model_path.stat().st_size <= 1024 * 1024:
            print(f"[TTS] Downloading Silero model directly: {model_url}")
            print(f"[TTS] Target: {model_path}")
            torch.hub.download_url_to_file(model_url, str(model_path), progress=True)

        print(f"[TTS] Loading Silero direct package after download: {model_path}")
        return self._try_direct_package(model_path)

    def generate(self, text: str) -> np.ndarray:
        text = _clean_text(text)
        if not text:
            return np.zeros(1, dtype=np.float32)

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
    model_id = model_id or os.environ.get("SILERO_MODEL", "v4_ru")
    speaker = speaker or os.environ.get("SILERO_SPEAKER", os.environ.get("TTS_VOICE", "baya"))
    sample_rate = int(sample_rate or os.environ.get("SILERO_SAMPLE_RATE", "24000"))
    speed = float(speed if speed is not None else os.environ.get("SILERO_SPEED", os.environ.get("TTS_SPEED", "1.0")))
    return SileroBackend(model_id=model_id, speaker=speaker, sample_rate=sample_rate, speed=speed)
