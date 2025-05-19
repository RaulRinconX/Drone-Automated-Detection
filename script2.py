#!/usr/bin/env python3
"""
detectar_dron_v2.py
Detección robusta de dron con umbral adaptado, filtro de banda y
histeresis temporal.

Uso:
    python detectar_dron_v2.py baseline.csv data.csv [opciones]

Posicionales
------------
baseline.csv : mediciones con el dron APAGADO (calibra umbrales)
data.csv     : mediciones a evaluar (detección)

Opciones principales
--------------------
-k KSIGMA          Factor σ para el umbral (default 3.0)
--freq-min FMIN    Frecuencia mínima a vigilar (MHz)
--freq-max FMAX    Frecuencia máxima a vigilar (MHz)
--n-consec N       Nº de muestras consecutivas requeridas (default 1)
--stats            Imprime estadísticas y umbrales empleados
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------- PARSING DE ARGUMENTOS ------------------------------ #
parser = argparse.ArgumentParser(description="Detección de dron con mitigación de falsos positivos/negativos.")
parser.add_argument("baseline", help="CSV con el dron apagado.")
parser.add_argument("datafile", help="CSV con mediciones a analizar.")
parser.add_argument("-k", "--ksigma", type=float, default=3.0,
                    help="Multiplicador de la desviación estándar para el umbral (default 3.0).")
parser.add_argument("--freq-min", type=float, default=None,
                    help="Frecuencia mínima (MHz) para considerar la detección.")
parser.add_argument("--freq-max", type=float, default=None,
                    help="Frecuencia máxima (MHz) para considerar la detección.")
parser.add_argument("--n-consec", type=int, default=1,
                    help="Número de muestras consecutivas sobre umbral necesarias (default 1).")
parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas y umbrales.")
args = parser.parse_args()

# ------------------------ FUNCIONES AUXILIARES ----------------------------- #
def leer_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"ERROR: Archivo inexistente: {path}")
        sys.exit(1)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def buscar_col(df: pd.DataFrame, key: str):
    for c in df.columns:
        if key.lower() in c.lower():
            return c
    return None

# ------------------------ CARGA DE ARCHIVOS -------------------------------- #
df_base = leer_csv(args.baseline)
df_data = leer_csv(args.datafile)

pfd_base = buscar_col(df_base, "Power Flux Density")
pfd_data = buscar_col(df_data, "Power Flux Density")
pow_base = buscar_col(df_base, "Total Spectrum Power")
pow_data = buscar_col(df_data, "Total Spectrum Power")
freq_data = buscar_col(df_data, "Frequency")  # solo para filtro de banda

if not pfd_base or not pfd_data:
    print("ERROR: No se encontró 'Power Flux Density' en uno de los archivos.")
    sys.exit(1)

# ---------------------- CÁLCULO DE UMBRALES -------------------------------- #
mu_pfd,  sigma_pfd  = df_base[pfd_base].mean(), df_base[pfd_base].std()
thr_pfd = mu_pfd + args.ksigma * sigma_pfd

thr_power = None
if pow_base and pow_data:
    mu_pow, sigma_pow = df_base[pow_base].mean(), df_base[pow_base].std()
    thr_power = mu_pow + args.ksigma * sigma_pow

# ------------------------- FILTRO DE BANDA --------------------------------- #
band_mask = np.ones(len(df_data), dtype=bool)
if freq_data and args.freq_min is not None and args.freq_max is not None:
    band_mask = (df_data[freq_data] >= args.freq_min) & (df_data[freq_data] <= args.freq_max)

# ----------------------- DETECCIÓN INSTANTÁNEA ----------------------------- #
instant_mask = (df_data[pfd_data] > thr_pfd) & band_mask
if thr_power is not None:
    instant_mask &= df_data[pow_data] > thr_power

# ---------------------------- HISTERESIS ----------------------------------- #
if args.n_consec > 1:
    # Rolling suma: True=1, False=0 -> si la ventana suma >= n_consec => condición cumplida
    hits = pd.Series(instant_mask.astype(int)).rolling(args.n_consec, min_periods=args.n_consec).sum() >= args.n_consec
    dron_detectado = hits.any()
else:
    dron_detectado = instant_mask.any()

# --------------------------- SALIDAS --------------------------------------- #
if args.stats:
    print("\n--- BASELINE ---")
    print(f"{pfd_base}: μ={mu_pfd:.2f}, σ={sigma_pfd:.2f}  ->  umbral={thr_pfd:.2f}")
    if thr_power is not None:
        print(f"{pow_base}: μ={mu_pow:.2f} dBm, σ={sigma_pow:.2f}  ->  umbral={thr_power:.2f} dBm")
    if args.freq_min is not None:
        print(f"Filtro de banda: {args.freq_min}–{args.freq_max} MHz")
    print("\n--- DATA describe() ---")
    cols_show = [c for c in [pfd_data, pow_data, freq_data] if c]
    print(df_data[cols_show].describe().T)

print("\n===== RESULTADO =====")
print("Drone detectado" if dron_detectado else "Sin drone")
