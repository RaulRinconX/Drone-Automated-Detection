#!/usr/bin/env python3
"""
detectar_dron_v3.py
Robust drone detection using adaptive thresholding in PFD (dB) and peak (dBm),
bandpass filter, and temporal hysteresis.

Usage:
    python detectar_dron_v3.py baseline.csv data.csv [options]
"""
import argparse
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import Jetson.GPIO as GPIO


def leer_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"ERROR: File does not exist: {path}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Drone detection using thresholds in PFD and peak, bandpass filter, and hysteresis."
    )
    parser.add_argument("baseline", help="Processed CSV (drone off).")
    parser.add_argument("datafile", help="Processed CSV to analyze.")
    parser.add_argument("-k", "--ksigma", type=float, default=3.0,
                        help="Multiplier for σ to calculate thresholds (default 3.0).")
    parser.add_argument("--freq-min", type=float, default=None,
                        help="Minimum frequency (MHz) for detection.")
    parser.add_argument("--freq-max", type=float, default=None,
                        help="Maximum frequency (MHz) for detection.")
    parser.add_argument("--n-consec", type=int, default=1,
                        help="Required consecutive samples (default 1).")
    parser.add_argument("--stats", action="store_true",
                        help="Display statistics and thresholds.")
    args = parser.parse_args()

    # 1) Load both CSV files
    df_base = leer_csv(args.baseline)
    df_data = leer_csv(args.datafile)

    # 2) Find relevant columns
    pfd_base_col   = buscar_col(df_base, "Power Flux Density")
    pfd_data_col   = buscar_col(df_data, "Power Flux Density")
    peak_base_col  = buscar_col(df_base, "Max Amplitude")
    peak_data_col  = buscar_col(df_data, "Max Amplitude")
    freq_col       = buscar_col(df_data, "Frequency")

    if not (pfd_base_col and pfd_data_col and peak_base_col and peak_data_col):
        print("ERROR: Could not find PFD or peak columns in the files.")
        sys.exit(1)

    # 3) Convert to float and clean zeros/NaN before log
    eps = 1e-12
    df_base["pfd_lin"] = df_base[pfd_base_col].astype(float).clip(lower=eps)
    df_data["pfd_lin"] = df_data[pfd_data_col].astype(float).clip(lower=eps)

    df_base["pfd_dB"]  = 10 * np.log10(df_base["pfd_lin"])
    df_data["pfd_dB"]  = 10 * np.log10(df_data["pfd_lin"])

    df_base["peak_dB"] = df_base[peak_base_col].astype(float)
    df_data["peak_dB"] = df_data[peak_data_col].astype(float)

    # 4) Remove inf/nan for the baseline
    clean_base = df_base.replace([np.inf, -np.inf], np.nan).dropna(subset=["pfd_dB", "peak_dB"])

    # 5) Statistics and thresholds
    mu_pfd,  sig_pfd  = clean_base["pfd_dB"].mean(), clean_base["pfd_dB"].std()
    thr_pfd             = mu_pfd + args.ksigma * sig_pfd

    mu_peak, sig_peak = clean_base["peak_dB"].mean(), clean_base["peak_dB"].std()
    thr_peak            = mu_peak + args.ksigma * sig_peak

    # 6) Bandpass filter (if applicable)
    mask_band = np.ones(len(df_data), dtype=bool)
    if freq_col and args.freq_min is not None and args.freq_max is not None:
        mask_band = (
            (df_data[freq_col] >= args.freq_min) &
            (df_data[freq_col] <= args.freq_max)
        )

    # 7) Instantaneous detection: PFD>dB and peak>dBm
    inst = (
        (df_data["pfd_dB"] > thr_pfd) &
        (df_data["peak_dB"] > thr_peak) &
        mask_band
    )

    # 8) Temporal hysteresis
    if args.n_consec > 1:
        hits = (
            pd.Series(inst.astype(int))
            .rolling(args.n_consec, min_periods=args.n_consec)
            .sum() >= args.n_consec
        )
        detected = hits.any()
    else:
        detected = inst.any()

    # 9) Logging outputs
    if args.stats:
        print(f"\n--- BASELINE ---")
        print(f"PFD dB:   μ={mu_pfd:.2f}  σ={sig_pfd:.2f}  → threshold={thr_pfd:.2f} dB")
        print(f"Peak dBm: μ={mu_peak:.2f}  σ={sig_peak:.2f}  → threshold={thr_peak:.2f} dBm")
        if args.freq_min is not None:
            print(f"Band: {args.freq_min}–{args.freq_max} MHz")
        print("\n--- DATA describe() ---")
        cols_show = ["pfd_dB", "peak_dB"]
        if freq_col: cols_show.append(freq_col)
        print(df_data[cols_show].describe().T)

    print("\n===== RESULT =====")
    if detected:
        print("Drone detected! Activating jammer…")
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(12, GPIO.OUT)
        GPIO.output(12, GPIO.HIGH)
        time.sleep(75)
        GPIO.output(12, GPIO.LOW)
    else:
        print("X No drone detected X")

    GPIO.cleanup()


if __name__ == "__main__":
    main()
