"""augmentator_v1_refactor.py (rev-5)
────────────────────────────────────────────────
Amplía la generación de segmentos con **ocho** técnicas clásicas de
`audio data‑augmentation`:

1. time_stretch       (±10 %)
2. pitch_shift        (±2 semitonos)
3. add_white_noise    (SNR = 15 dB)
4. polarity_invert    (‑1 · señal)
5. low_pass_filter    (3–6 kHz)
6. high_pass_filter   (200–600 Hz)
7. volume_perturb     (‑6 dB ↔ +6 dB)
8. reverb_simple      (convolución con IR sintética)

Por cada segmento base se exporta **el original + las 8 variantes** como WAV
(PCM 16 bit) con sufijos `_orig`, `_stretch`, `_pitch`, …

Requisitos extra:
  pip install numpy pandas pydub librosa scipy

FFmpeg debe seguir disponible en `C:/ffmpeg/bin`.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import os
import sys
import random
import numpy as np
import pandas as pd
from pydub import AudioSegment, effects, utils
from scipy.signal import fftconvolve
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# ⚙️  Configuración FFmpeg
# ---------------------------------------------------------------------------
FFMPEG_DIR = Path(r"C:/ffmpeg/bin")
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_EXE = FFMPEG_DIR / "ffprobe.exe"

os.environ["PATH"] += os.pathsep + str(FFMPEG_DIR)
if not (FFMPEG_EXE.is_file() and FFPROBE_EXE.is_file()):
    sys.exit("⚠️  ffmpeg/ffprobe no encontrados. Ajusta FFMPEG_DIR.")
AudioSegment.converter = str(FFMPEG_EXE)
AudioSegment.ffprobe = str(FFPROBE_EXE)

# ---------------------------------------------------------------------------
# 🖼️  Diálogos Tkinter
# ---------------------------------------------------------------------------

def seleccionar_audio() -> str:
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="Archivo de audio", filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"), ("Todos", "*.*")])
    root.destroy(); return path

def seleccionar_carpeta() -> str:
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="Carpeta destino")
    root.destroy(); return path

# ---------------------------------------------------------------------------
# 🎵 Clase AudioItem
# ---------------------------------------------------------------------------
class AudioItem:
    def __init__(self, path: str | Path, t0: datetime, label: str | None = None):
        self.path = Path(path)
        seg = AudioSegment.from_file(self.path)
        self.sr = seg.frame_rate
        self.channels = seg.channels
        self.duration_ms = len(seg)
        pcm = np.array(seg.get_array_of_samples())
        if self.channels > 1: pcm = pcm.reshape(-1, self.channels)
        else: pcm = pcm[:, None]
        self.data = pcm.astype(np.float32) / 32768.0
        self.t0 = t0
        self.label = label or self.path.stem

    # ------------------------------------------------------------------
    def split(self, slice_ms: int, *, include_reverse: bool = False) -> List["AudioItem"]:
        n_full = self.duration_ms // slice_ms
        return [self._clip_at(i * slice_ms, slice_ms) for i in range(n_full)]

    def _clip_at(self, start_ms: int, slice_ms: int) -> "AudioItem":
        n_samp = int(slice_ms * self.sr / 1000)
        idx = int(start_ms * self.sr / 1000)
        data = self.data[idx: idx + n_samp, :]
        child: AudioItem = object.__new__(AudioItem)
        child.path = self.path; child.sr = self.sr; child.channels = self.channels
        child.duration_ms = slice_ms; child.data = data; child.t0 = self.t0 + timedelta(milliseconds=start_ms)
        child.label = f"{self.label}_orig"
        return child

    # ------------------------------------------------------------------
    def to_dataframe_8bit(self) -> pd.DataFrame:
        arr_u8 = np.clip(((self.data + 1.0) * 127.5), 0, 255).astype(np.uint8)
        cols = [f"ch{i}" for i in range(self.channels)]
        return pd.DataFrame(arr_u8, columns=cols)

    # ------------------------------------------------------------------
    #  Augmentation helpers (static)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_mono_float32(seg: "AudioItem") -> np.ndarray:
        return seg.data.mean(axis=1).astype(np.float32)

# ---------------------------------------------------------------------------
# 🛠️  Funciones de augmentación (devuelven np.ndarray float32 mono)
# ---------------------------------------------------------------------------

def time_stretch(x: np.ndarray, sr: int, rate: float | None = None) -> np.ndarray:
    """Estira/encoge dinámicamente ±10 %. `sr` se ignora (solo para firma uniforme)."""
    rate = rate or random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(x, rate=rate)

def pitch_shift(x: np.ndarray, sr: int, steps: int | None = None) -> np.ndarray:
    steps = steps if steps is not None else random.choice([-2, -1, 1, 2])
    # En librosa ≥0.10 `sr` es keyword‑only → pásalo explícito
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)

def add_white_noise(x: np.ndarray, sr: int, snr_db: float = 15.0) -> np.ndarray:
    """Añade ruido blanco a una SNR deseada; robusto a señales casi silenciosas."""
    eps = 1e-12  # evita división por cero
    sig_power = float(np.mean(x.astype(np.float64) ** 2))
    sig_power = max(sig_power, eps)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return x + noise

def polarity_invert(x: np.ndarray, sr: int, **_) -> np.ndarray:
    """Invierte la polaridad (‑1 · señal). `sr` se ignora."""
    return -x

from scipy.signal import butter, filtfilt

def _butter_filter(x: np.ndarray, sr: int, cutoff: int, btype: str) -> np.ndarray:
    nyq = 0.5 * sr
    norm = cutoff / nyq
    b, a = butter(4, norm, btype=btype)
    return filtfilt(b, a, x).astype(np.float32)

def low_pass_filter(signal: np.ndarray, sr: int, cutoff: int | None = None) -> np.ndarray:
    cutoff = cutoff or random.randint(3000, 6000)
    return _butter_filter(signal, sr, cutoff, "low")

def high_pass_filter(signal: np.ndarray, sr: int, cutoff: int | None = None) -> np.ndarray:
    cutoff = cutoff or random.randint(200, 600)
    return _butter_filter(signal, sr, cutoff, "high")

def volume_perturb(signal: np.ndarray, sr: int, gain_db: float | None = None) -> np.ndarray:
    """Ajusta el volumen ±6 dB (por defecto) y normaliza."""
    gain_db = gain_db if gain_db is not None else random.uniform(-6, 6)
    return librosa.util.normalize(signal * (10 ** (gain_db / 20)))

def reverb_simple(signal: np.ndarray, sr: int) -> np.ndarray:
    """Añade una reverberación sintética por convolución con un IR exponencial."""
    decay = np.exp(-np.linspace(0, 3, int(sr * 0.3)))
    ir = np.concatenate(([1.0], decay))
    wet = fftconvolve(signal, ir, mode="full")[: len(signal)]
    return librosa.util.normalize(wet)

# Diccionario de nombre → función
# Diccionario de nombre → función (todas aceptan (x, sr) como args)
AUG_FUNCS: Dict[str, callable] = {
    "stretch": time_stretch,
    "pitch": pitch_shift,
    "noise": add_white_noise,
    "invert": polarity_invert,
    "lpf": low_pass_filter,
    "hpf": high_pass_filter,
    "volume": volume_perturb,
    "reverb": reverb_simple,
}

# ---------------------------------------------------------------------------
# 💾 Guardado WAV helper
# ---------------------------------------------------------------------------

def save_wav(mono: np.ndarray, sr: int, path: Path):
    pcm16 = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
    seg = AudioSegment(
        data=pcm16.tobytes(), sample_width=2, frame_rate=sr, channels=1
    )
    seg.export(path, format="wav")

# ---------------------------------------------------------------------------
# 🚀 Flujo principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    src = seleccionar_audio()
    if not src:
        sys.exit("Cancelado.")
    dst = seleccionar_carpeta()
    if not dst:
        sys.exit("Cancelado.")

    base = AudioItem(src, t0=datetime.now())
    segments = base.split(slice_ms=250)

    out_dir = Path(dst)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for idx, seg in enumerate(segments):
        mono = seg.data.mean(axis=1)
        # Guarda original
        save_wav(mono, seg.sr, out_dir / f"{seg.label}_{idx:04d}_orig.wav")
        total += 1
        # Aplica cada augmentación                           
        for suf, func in AUG_FUNCS.items():
            aug = func(mono, seg.sr)
            save_wav(aug, seg.sr, out_dir / f"{seg.label}_{idx:04d}_{suf}.wav")
            total += 1

    messagebox.showinfo("Fin", f"Se crearon {total} archivos en\n{out_dir}")
