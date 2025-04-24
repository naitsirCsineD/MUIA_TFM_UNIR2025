import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox




def seleccionar_carpeta() -> str:
    root = tk.Tk(); root.withdraw()
    path = filedialog.askdirectory(title="Carpeta destino")
    root.destroy(); return path

class AudioValidator:
    def __init__(self, carpeta_audio, sr=16000, min_duracion=0.25, min_snr_db=5):
        self.carpeta = carpeta_audio
        self.sr = sr
        self.min_duracion = min_duracion
        self.min_snr_db = min_snr_db
        self.datos_validos = []

    def calcular_snr(self, señal):
        potencia_total = np.mean(señal**2)
        potencia_ruido = np.mean((señal - np.mean(señal))**2)
        if potencia_ruido == 0:
            return 100
        return 10 * np.log10(potencia_total / potencia_ruido)

    def es_valido(self, señal):
        duracion = len(señal) / self.sr
        if duracion < self.min_duracion:
            return False
        
        snr = self.calcular_snr(señal)
        if snr < self.min_snr_db:
            return False

        if np.allclose(señal, 0) or np.std(señal) == 0:
            return False

        if np.max(np.abs(señal)) > 1.0:
            return False

        return True

    def procesar_archivos(self):
        for archivo in os.listdir(self.carpeta):
            if not archivo.endswith(".wav"):
                continue
            ruta = os.path.join(self.carpeta, archivo)
            try:
                señal, _ = librosa.load(ruta, sr=self.sr, mono=True)
                if self.es_valido(señal):
                    señal_8bytes = señal.astype(np.float64)
                    tiempo = np.arange(len(señal)) / self.sr  # tiempo por muestra
                    for i, (valor, t) in enumerate(zip(señal_8bytes, tiempo)):
                        self.datos_validos.append({
                            'archivo': archivo,
                            'índice': i,
                            'tiempo': t,
                            'valor': valor
                        })
            except Exception as e:
                print(f"Error al procesar {archivo}: {e}")
        
        return pd.DataFrame(self.datos_validos)

# === USO ===
validador = AudioValidator(seleccionar_carpeta())
df_valido = validador.procesar_archivos()
print(df_valido.head())
# Guardar DataFrame en CSV
df_valido.to_csv("C:\\MUIA2024\\MUIA,TFM\\Archivos de Audio\\datos_validos.csv", index=False)