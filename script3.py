#!/usr/bin/env python3
"""
detectar_dron_v3.py
Detección robusta de dron con umbral adaptado en dB, filtro de banda y
histeresis temporal.

Uso:
    python detectar_dron_v3.py baseline.csv data.csv [opciones]

Posicionales
------------
baseline.csv : mediciones con el dron APAGADO (calibra umbrales)
data.csv     : mediciones a evaluar (detección)

Opciones principales
--------------------
-k KSIGMA          Factor σ para el umbral (default 3.0)
--freq-min FMIN    Frecuencia mínima a vigilar (MHz)
--freq-max FMAX    Frecuencia máxima a vigilar (MHz)
--n-consec N       Nº de muestras consecutivas sobre umbral necesarias (default 1)
--stats            Imprime estadísticas y umbrales empleados
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import Jetson.GPIO as GPIO
import time


def leer_csv(path: str) -> pd.DataFrame:
    """Carga un CSV, probando utf-8 y latin1."""
    path = Path(path)
    if not path.exists():
        print(f"ERROR: Archivo inexistente: {path}")
        sys.exit(1)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def buscar_col(df: pd.DataFrame, key: str):
    """Busca columna cuyo nombre contenga (case-insensitive) la llave."""
    for c in df.columns:
        if key.lower() in c.lower():
            return c
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Detección de dron con umbral en dB, filtro de banda e histeresis."
    )
    parser.add_argument("baseline", help="CSV con el dron apagado.")
    parser.add_argument("datafile", help="CSV con mediciones a analizar.")
    parser.add_argument("-k", "--ksigma", type=float, default=3.0,
                        help="Multiplicador de σ para el umbral (default 3.0).")
    parser.add_argument("--freq-min", type=float, default=None,
                        help="Frecuencia mínima (MHz) para detección.")
    parser.add_argument("--freq-max", type=float, default=None,
                        help="Frecuencia máxima (MHz) para detección.")
    parser.add_argument("--n-consec", type=int, default=1,
                        help="Muestras consecutivas requeridas (default 1).")
    parser.add_argument("--stats", action="store_true",
                        help="Mostrar estadísticas y umbrales.")
    args = parser.parse_args()

    # Carga de datos
    df_base = leer_csv(args.baseline)
    df_data = leer_csv(args.datafile)

    # Columnas relevantes
    pfd_base_col = buscar_col(df_base, "Power Flux Density")
    pfd_data_col = buscar_col(df_data, "Power Flux Density")
    pow_base_col = buscar_col(df_base, "Total Spectrum Power")
    pow_data_col = buscar_col(df_data, "Total Spectrum Power")
    freq_data_col = buscar_col(df_data, "Frequency")

    if not pfd_base_col or not pfd_data_col:
        print("ERROR: No se encontró 'Power Flux Density' en uno de los archivos.")
        sys.exit(1)

    # 1) Convertir Power Flux Density a dB
    df_base["pfd_dB"] = 10 * np.log10(df_base[pfd_base_col].replace(0, np.nan))
    df_data["pfd_dB"] = 10 * np.log10(df_data[pfd_data_col].replace(0, np.nan))

    # 2) Calcular umbral en dB
    mu_db = df_base["pfd_dB"].mean()
    sigma_db = df_base["pfd_dB"].std()
    thr_db = mu_db + args.ksigma * sigma_db

    # Si usamos también Total Spectrum Power:
    thr_pow = None
    if pow_base_col and pow_data_col:
        mu_pow = df_base[pow_base_col].mean()
        sigma_pow = df_base[pow_base_col].std()
        thr_pow = mu_pow + args.ksigma * sigma_pow

    # 3) Filtro de banda (opcional)
    band_mask = np.ones(len(df_data), dtype=bool)
    if freq_data_col and args.freq_min is not None and args.freq_max is not None:
        band_mask = (
            (df_data[freq_data_col] >= args.freq_min) &
            (df_data[freq_data_col] <= args.freq_max)
        )

    # 4) Detección instantánea: pfd_dB > thr_db
    instant_mask = (df_data["pfd_dB"] > thr_db) & band_mask
    if thr_pow is not None:
        instant_mask &= (df_data[pow_data_col] > thr_pow)

    # 5) Histeresis: n_consec muestras seguidas
    if args.n_consec > 1:
        hits = (
            pd.Series(instant_mask.astype(int))
            .rolling(args.n_consec, min_periods=args.n_consec)
            .sum() >= args.n_consec
        )
        dron_detectado = hits.any()
    else:
        dron_detectado = instant_mask.any()

    # 6) Salidas
    if args.stats:
        print("\n--- BASELINE (PFD en dB) ---")
        print(f"PFD dB: μ={mu_db:.2f}, σ={sigma_db:.2f}  ->  umbral={thr_db:.2f} dB")
        if thr_pow is not None:
            print(f"{pow_base_col}: μ={mu_pow:.2f} dBm, σ={sigma_pow:.2f}  ->  umbral={thr_pow:.2f} dBm")
        if args.freq_min is not None:
            print(f"Banda usada: {args.freq_min}–{args.freq_max} MHz")
        print("\n--- DATA describe() ---")
        cols_show = ["pfd_dB", pow_data_col, freq_data_col]
        print(df_data[cols_show].describe().T)

    print("\n===== RESULTADO =====")
    print("Drone detectado" if dron_detectado else "Sin drone")
    
    if dron_detectado:
        print("Drone detected! Starting jammer...")
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(12, GPIO.OUT)
        GPIO.output(12, GPIO.HIGH)
        time.sleep(75)
        GPIO.output(12, GPIO.LOW)


if __name__ == "__main__":
    main()
