import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import time
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram


def procesar_wav_a_csv(
    ruta_wav: Path,
    out_dir: Path,
    ventana_ms: float,
    solap: float,
    min_hz: int,
    max_hz: int,
    step_hz: int
):
    """
    Procesa un archivo WAV para extraer PSD en ventanas y guarda un CSV con:
    - timestamp: tiempo del centro de la ventana (s)
    - dB_<hz>Hz: intensidad en cada frecuencia en dB/Hz
    - amplitud_max: valor absoluto máximo de la señal en la ventana
    """
    datos, fs = sf.read(ruta_wav)
    if datos.ndim > 1:
        datos = np.mean(datos, axis=1)

    segment_samples = int((ventana_ms / 1000) * fs)
    hop = max(1, int(segment_samples * (1 - solap)))
    freqs = list(range(min_hz, max_hz + 1, step_hz))

    filas = []
    for start in range(0, len(datos) - segment_samples + 1, hop):
        bloque = datos[start:start + segment_samples]
        t_cent = (start + segment_samples / 2) / fs

        amp_max = float(np.max(np.abs(bloque)))

        nperseg = segment_samples
        noverlap = int(segment_samples * solap)
        f_psd, _, Sxx = spectrogram(
            bloque,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            mode="psd"
        )
        Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

        intens = {f"dB_{hz}Hz": np.interp(hz, f_psd, Sxx_db[:, 0]) for hz in freqs}

        fila = {"timestamp": t_cent, "amplitud_max": amp_max}
        fila.update(intens)
        filas.append(fila)

    df = pd.DataFrame(filas)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{ruta_wav.stem}_features.csv"
    df.to_csv(csv_path, index=False)


def main_gui():
    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("WAV → CSV: Espectrograma + Amplitud")
            self.geometry("480x420")
            self.resizable(False, False)

            # Variables
            self.input_var = tk.StringVar()
            self.output_var = tk.StringVar()
            self.ventana_var = tk.StringVar(value="100")
            self.solap_var = tk.StringVar(value="0.5")
            self.min_hz_var = tk.StringVar(value="1")
            self.max_hz_var = tk.StringVar(value="20")
            self.step_hz_var = tk.StringVar(value="1")

            # Selección de carpetas
            ttk.Label(self, text="Carpeta WAV:").pack(pady=(10, 0), anchor="w", padx=10)
            frame1 = ttk.Frame(self); frame1.pack(fill="x", padx=10)
            ttk.Entry(frame1, textvariable=self.input_var).pack(side="left", fill="x", expand=True)
            ttk.Button(frame1, text="…", width=3, command=self.select_input).pack(side="left", padx=5)

            ttk.Label(self, text="Carpeta salida CSV:").pack(pady=5, anchor="w", padx=10)
            frame2 = ttk.Frame(self); frame2.pack(fill="x", padx=10)
            ttk.Entry(frame2, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
            ttk.Button(frame2, text="…", width=3, command=self.select_output).pack(side="left", padx=5)

            # Parámetros con Combobox
            params = [
                ("Ventana (ms):", self.ventana_var, ["50","100","200","500","1000"]),
                ("Solapamiento:", self.solap_var, ["0.25","0.5","0.75"]),
                ("Min Hz:", self.min_hz_var, [str(i) for i in range(1,11)]),
                ("Max Hz:", self.max_hz_var, [str(i) for i in range(10,101,10)]),
                ("Step Hz:", self.step_hz_var, ["1","2","5","10"]),
            ]
            for label, var, values in params:
                frame = ttk.Frame(self); frame.pack(fill="x", padx=10, pady=5)
                ttk.Label(frame, text=label, width=15).pack(side="left")
                cb = ttk.Combobox(frame, textvariable=var, values=values, width=10)
                cb.pack(side="left")

            # Progress bar
            self.progress = ttk.Progressbar(self, orient="horizontal", length=460, mode="determinate")
            self.progress.pack(pady=(10,5), padx=10)

            # Botón de ejecución y estado
            ttk.Button(self, text="Procesar carpeta", command=self.run).pack(pady=(5,10))
            self.status = ttk.Label(self, text="", foreground="green")
            self.status.pack(pady=(0,10))

        def select_input(self):
            d = filedialog.askdirectory(title="Selecciona carpeta WAV")
            if d: self.input_var.set(d)

        def select_output(self):
            d = filedialog.askdirectory(title="Selecciona carpeta salida CSV")
            if d: self.output_var.set(d)

        def run(self):
            input_dir = Path(self.input_var.get()); output_dir = Path(self.output_var.get())
            if not input_dir.is_dir() or not output_dir.is_dir():
                messagebox.showwarning("Error", "Selecciona carpetas válidas.")
                return

            try:
                ventana = float(self.ventana_var.get()); solap = float(self.solap_var.get())
                min_hz = int(self.min_hz_var.get()); max_hz = int(self.max_hz_var.get())
                step_hz = int(self.step_hz_var.get())
            except ValueError:
                messagebox.showwarning("Error", "Parámetros inválidos.")
                return

            wavs = sorted(Path(input_dir).glob("*.wav"))
            total = len(wavs)
            if total == 0:
                messagebox.showinfo("Info", "No se encontraron .wav en la carpeta.")
                return

            self.progress['maximum'] = total
            start_time = time.perf_counter()

            for idx, wav in enumerate(wavs, 1):
                try:
                    procesar_wav_a_csv(
                        wav, output_dir,
                        ventana_ms=ventana, solap=solap,
                        min_hz=min_hz, max_hz=max_hz, step_hz=step_hz
                    )
                except Exception as e:
                    print(f"Error en {wav.name}: {e}")

                # Actualizar progreso
                self.progress['value'] = idx
                percent = idx / total * 100
                elapsed = time.perf_counter() - start_time
                remaining = elapsed / idx * (total - idx)
                self.status.config(text=f"{percent:.1f}% completado - ETA: {remaining:.1f}s")
                self.update_idletasks()

            elapsed = time.perf_counter() - start_time
            messagebox.showinfo(
                "Completado",
                f"Procesados {total} archivos en {elapsed:.1f}s."
            )
            self.destroy()

    app = App()
    app.mainloop()

if __name__ == "__main__":
    main_gui()
