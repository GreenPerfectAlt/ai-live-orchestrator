"""Standalone Silero TTS HTTP server for xiaozhi-server integration."""

import io
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

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
    text = text.replace("\u2014", "-")
    text = SPACE_RE.sub(" ", text).strip()
    if text and text[-1] not in ".!?...,:;":
        text += "."
    return text


def _as_float32_mono(audio: Any) -> np.ndarray:
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
        arr = arr / peak
    return np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)


def _normalize_speaker(speaker: str) -> str:
    raw = str(speaker or "").strip().lower()
    mapping = {
        "f4": "baya",
        "f3": "xenia",
        "female": "xenia",
        "male": "aidar",
    }
    raw = mapping.get(raw, raw)
    allowed = {"aidar", "baya", "kseniya", "xenia", "eugene", "random"}
    return raw if raw in allowed else "xenia"


class SileroBackend:
    def __init__(
        self,
        model_id: str = "v5_5_ru",
        speaker: str = "xenia",
        sample_rate: int = 24000,
    ) -> None:
        self.model_id = model_id or "v5_5_ru"
        self.speaker = _normalize_speaker(speaker)
        self.sample_rate = int(sample_rate or 24000)
        if self.sample_rate not in {8000, 24000, 48000}:
            self.sample_rate = 24000
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

        try:
            self.generate("Привет.")
        except Exception as exc:
            print(f"[TTS] Silero warmup warning: {exc}")

        print(
            f"✅ TTS: Silero RU, model={self.model_id}, speaker={self.speaker}, "
            f"sample_rate={self.sample_rate}"
        )

    def _import_torch(self):
        import torch

        threads = int(os.environ.get("TTS_THREADS", "4"))
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

    def _load_model(self):
        torch = self._torch
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)

        model_path = Path(os.environ.get("SILERO_MODEL_PATH", str(cache / f"{self.model_id}.pt"))).expanduser()

        if model_path.exists() and model_path.stat().st_size > 1024 * 1024:
            print(f"[TTS] Loading Silero local package: {model_path}")
            return self._try_direct_package(model_path)
        else:
            raise FileNotFoundError(f"Model file not found or too small: {model_path}")

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
        return arr


# Initialize backend
backend = SileroBackend(model_id="v5_5_ru", speaker="xenia", sample_rate=24000)

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "silero-v5_5_ru"}


@app.post("/v1/audio/speech")
async def tts(request: Request):
    """OpenAI-compatible TTS endpoint for xiaozhi-server."""
    data = await request.json()
    text = data.get("input", "")
    speaker = data.get("voice", "xenia")
    
    if not text:
        return {"error": "No text provided"}
    
    print(f"[TTS] Generating speech: {text[:50]}...")
    
    # Generate audio
    audio = backend.generate(text)
    
    # Convert to WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, backend.sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    
    return StreamingResponse(buffer, media_type="audio/wav")


if __name__ == "__main__":
    print("🚀 Starting Silero TTS server on http://127.0.0.1:8880")
    print("   Endpoint: POST /v1/audio/speech")
    print("   Health: GET /health")
    uvicorn.run(app, host="127.0.0.1", port=8880, log_level="info")