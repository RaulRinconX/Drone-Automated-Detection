#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

def process_with_pandas(input_csv: str, output_csv: str, flux_cal: float):
    # 1) Leer CSV raw sin encabezado
    df = pd.read_csv(input_csv, header=None)

    # 2) Asignar nombres: las primeras 6 fijas y el resto "db1", "db2", ...
    col_count = df.shape[1]
    base_cols = ['date', 'time', 'hz_low', 'hz_high', 'hz_bin_width', 'num_samples']
    db_cols = [f'db{i+1}' for i in range(col_count - len(base_cols))]
    df.columns = base_cols + db_cols

    # 3) Combinar date + time en Timestamp (sin decimales de segundo)
    df['Timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'].str.split('.').str[0],
        format='%Y-%m-%d %H:%M:%S'
    ).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 4) Pasar de dB a mW/Hz:   P_mW/Hz = 10^(dBm/10)
    lin_mw_per_hz = 10 ** (df[db_cols] / 10)

    # 5) Sumar a lo largo de todas las bandas (cada fila)
    total_lin_mw_per_hz = lin_mw_per_hz.sum(axis=1)

    # 6) Convertir a mW sobre todo el ancho de banda (bin_width en Hz)
    total_mw = total_lin_mw_per_hz * df['hz_bin_width']

    # 7) Calcular Power Flux Density [µW/m²]:
    #    total_mw (mW) -> µW: *1000, luego aplicar factor de calibración
    df['Power Flux Density [µW/m²]'] = total_mw * 1000 * flux_cal

    # 8) Total Spectrum Power [dBm]: 10·log10(total_mw)
    #    Evitamos log(0) reemplazándolo por NaN y luego llenamos con un valor muy bajo
    df['Total Spectrum Power [dBm]'] = 10 * np.log10(total_mw.replace(0, np.nan))
    df['Total Spectrum Power [dBm]'] = df['Total Spectrum Power [dBm]'].fillna(-150)

    # 9) Max Amplitude [dBm] y Frequency [MHz]
    df['Max Amplitude [dBm]'] = df[db_cols].max(axis=1)
    idx_max = df[db_cols].values.argmax(axis=1)
    # Frecuencia central de cada bin: hz_low + (i+0.5)*hz_bin_width
    df['Frequency [MHz]'] = (
        df['hz_low'] + (idx_max + 0.5) * df['hz_bin_width']
    ) / 1e6

    # 10) Formatear sin notación científica y con decimales fijos
    df['Total Spectrum Power [dBm]']    = df['Total Spectrum Power [dBm]'].map(lambda x: f"{x:.1f}")
    df['Power Flux Density [µW/m²]']    = df['Power Flux Density [µW/m²]'].map(lambda x: f"{x:.1f}")
    df['Max Amplitude [dBm]']           = df['Max Amplitude [dBm]'].map(lambda x: f"{x:.1f}")
    df['Frequency [MHz]']               = df['Frequency [MHz]'].map(lambda x: f"{x:.2f}")

    # 11) Exportar sólo las columnas deseadas
    out = df[[
        'Timestamp',
        'Total Spectrum Power [dBm]',
        'Power Flux Density [µW/m²]',
        'Max Amplitude [dBm]',
        'Frequency [MHz]'
    ]]
    out.to_csv(output_csv, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Convierte sweep raw CSV a formato simplificado usando pandas"
    )
    parser.add_argument('-i', '--input',  required=True, help="CSV raw, sin encabezado")
    parser.add_argument('-o', '--output', required=True, help="CSV de salida")
    parser.add_argument('-f', '--flux-cal', type=float, default=50.0,
                        help="Factor de calibración (default: 50)")
    args = parser.parse_args()

    process_with_pandas(args.input, args.output, args.flux_cal)

if __name__ == "__main__":
    main()
