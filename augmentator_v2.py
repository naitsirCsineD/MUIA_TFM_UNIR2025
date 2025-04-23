"""augmentator_v1_refactor.py
------------------------------------------------
Utilidad para:
  1. Elegir un archivo de audio (Tkinter).
  2. Generar segmentos normales y espejo.
  3. Preguntar carpeta destino y guardar los segmentos en WAV.

Requisitos   : pip install pydub numpy
Dependencias : FFmpeg en C:/ffmpeg/bin o ruta equivalente.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import os
import sys
import numpy as np
from pydub import AudioSegment, utils
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# ⚙️  Configuración FFmpeg
# ---------------------------------------------------------------------------
FFMPEG_DIR = Path(r"C:/ffmpeg/bin")          # ⇦ ajusta si usas otra carpeta
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_EXE = FFMPEG_DIR / "ffprobe.exe"

os.environ["PATH"] += os.pathsep + str(FFMPEG_DIR)
if not (FFMPEG_EXE.is_file() and FFPROBE_EXE.is_file()):
    sys.exit(f"ffmpeg/ffprobe no encontrados en {FFMPEG_DIR}. Ajusta FFMPEG_DIR.")
AudioSegment.converter = str(FFMPEG_EXE)
AudioSegment.ffprobe   = str(FFPROBE_EXE)

# ---------------------------------------------------------------------------
# 🖼️  Diálogos Tkinter
# ---------------------------------------------------------------------------

def seleccionar_audio() -> str:
    """Devuelve la ruta al archivo elegido o '' si se cancela."""
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Seleccione archivo de audio",
        filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"), ("Todos", "*.*")],
    )
    root.destroy()
    return path

def seleccionar_carpeta() -> str:
    """Devuelve la carpeta de salida elegida o '' si se cancela."""
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="Carpeta destino para los segmentos")
    root.destroy()
    return path

# ---------------------------------------------------------------------------
# 🎵 Clase AudioItem
# ---------------------------------------------------------------------------
class AudioItem:
    def __init__(self, path: str | Path, t0: datetime, label: str | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        seg = AudioSegment.from_file(self.path)
        self.sr = seg.frame_rate
        self.channels = seg.channels
        self.duration_ms = len(seg)
        pcm = np.array(seg.get_array_of_samples())
        if self.channels > 1:
            pcm = pcm.reshape(-1, self.channels)
        else:
            pcm = pcm[:, None]
        self.data = pcm.astype(np.float32) / (2 ** (8 * seg.sample_width - 1))
        self.t0 = t0
        self.label = label or self.path.stem
        self.fmt = self.path.suffix.lstrip(".")

    def split(self, slice_ms: int) -> List["AudioItem"]:
        if slice_ms <= 0:
            raise ValueError("slice_ms debe ser > 0")
        total_ms = self.duration_ms
        n_full = total_ms // slice_ms
        n_samp = int(slice_ms * self.sr / 1000)
        childs: list[AudioItem] = []
        for i in range(n_full):
            dt = self.t0 + timedelta(milliseconds=i * slice_ms)
            data = self.data[i * n_samp : (i + 1) * n_samp, :]
            childs.append(_spawn_child(self, data, dt, slice_ms))
        for i in range(n_full):
            start_ms = total_ms - (i + 1) * slice_ms
            dt = self.t0 + timedelta(milliseconds=start_ms)
            idx = start_ms * self.sr // 1000
            data = self.data[int(idx) : int(idx) + n_samp, :][::-1]
            childs.append(_spawn_child(self, data, dt, slice_ms, mirrored=True))
        return childs

    def __repr__(self):
        return f"<AudioItem {self.label} • {self.duration_ms} ms • {self.sr} Hz • {self.channels} ch>"

# ---------------------------------------------------------------------------
# 🔧 Utilidad interna
# ---------------------------------------------------------------------------

def _spawn_child(parent: AudioItem, data: np.ndarray, t0: datetime, dur_ms: int, *, mirrored: bool = False) -> AudioItem:
    child: AudioItem = object.__new__(AudioItem)
    child.path = parent.path
    child.sr = parent.sr
    child.channels = parent.channels
    child.duration_ms = dur_ms
    child.data = data
    child.t0 = t0
    child.label = f"{parent.label}{'_rev' if mirrored else ''}"
    child.fmt = parent.fmt
    return child

# ---------------------------------------------------------------------------
# 💾 Guardado de segmentos
# ---------------------------------------------------------------------------

def export_segments(segments: List[AudioItem], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i, seg in enumerate(segments):
        fname = f"{seg.label}_{i:04d}.wav"
        out_path = Path(out_dir) / fname
        # NumPy → AudioSegment (16‑bit PCM mono/estéreo)
        raw = (seg.data * 32767).astype(np.int16)
        if seg.channels > 1:
            raw = raw.reshape(-1)
        pcm_bytes = raw.tobytes()
        wav = AudioSegment(
            data=pcm_bytes,
            sample_width=2,
            frame_rate=seg.sr,
            channels=seg.channels,
        )
        wav.export(out_path, format="wav")
    print(f"✔ Guardados {len(segments)} archivos en {out_dir}")

# ---------------------------------------------------------------------------
# 🚀 Ejecución interactiva
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    src = seleccionar_audio()
    if not src:
        sys.exit("Proceso cancelado. No se seleccionó audio.")
    dst = seleccionar_carpeta()
    if not dst:
        sys.exit("Proceso cancelado. No se seleccionó carpeta de salida.")

    original = AudioItem(src, t0=datetime.now())
    segs = original.split(1000)  # 250 ms
    export_segments(segs, dst)
    messagebox.showinfo("Fin", f"Se crearon {len(segs)} segmentos en\n{dst}")
