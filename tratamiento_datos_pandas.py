#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import math

def process_with_pandas(input_csv: str, output_csv: str, flux_cal: float):
    # 1) Leer CSV raw sin encabezado, saltando la primera línea
    df = pd.read_csv(
        input_csv,
        header=None,
        skiprows=1,
        skip_blank_lines=True,
        low_memory=False
    )

    # 2) Asignar nombres
    col_count = df.shape[1]
    base_cols = ['date', 'time', 'hz_low', 'hz_high', 'hz_bin_width', 'num_samples']
    db_cols = [f'db{i+1}' for i in range(col_count - len(base_cols))]
    df.columns = base_cols + db_cols

    # 3) Timestamp: unir date+time, quitar decimales y colapsar espacios
    ts = (df['date'].str.strip()
          + ' '
          + df['time'].str.strip().str.split('.').str[0])
    ts = ts.str.replace(r'\s+', ' ')
    df['Timestamp'] = pd.to_datetime(ts).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 4) dB -> mW/Hz -> mW/bin -> suma
    lin_mw_per_hz = 10 ** (df[db_cols] / 10)
    lin_mw_per_bin = lin_mw_per_hz.mul(df['hz_bin_width'], axis=0)
    power_sum_mw   = lin_mw_per_bin.sum(axis=1)

    # 5) Max Amplitude y Frequency
    df['Max Amplitude [dBm]'] = df[db_cols].max(axis=1)
    idx_max = df[db_cols].values.argmax(axis=1)
    df['Frequency [MHz]'] = (
        df['hz_low'] + (idx_max + 0.5) * df['hz_bin_width']
    ) / 1e6

    # 6) Power Flux Density según Java
    calibration_lin = 10 ** (flux_cal / 10.0)
    freq_mhz        = df['Frequency [MHz]']
    em_factor       = (4 * math.pi * (freq_mhz / 1e3) ** 2 * 1e18) / (299792458 ** 2)
    df['Power Flux Density [µW/m²]'] = power_sum_mw * calibration_lin * em_factor

    # 7) Total Spectrum Power [dBm]
    df['Total Spectrum Power [dBm]'] = 10 * np.log10(power_sum_mw.replace(0, np.nan))
    df['Total Spectrum Power [dBm]'].fillna(-150, inplace=True)

    # 8) Formatear salidas
    df['Total Spectrum Power [dBm]'] = df['Total Spectrum Power [dBm]'].map(lambda x: f"{x:.1f}")
    df['Power Flux Density [µW/m²]'] = df['Power Flux Density [µW/m²]'].map(lambda x: f"{x:.1f}")
    df['Max Amplitude [dBm]']        = df['Max Amplitude [dBm]'].map(lambda x: f"{x:.1f}")
    df['Frequency [MHz]']            = df['Frequency [MHz]'].map(lambda x: f"{x:.2f}")

    # 9) Exportar columnas finales
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
                        help="Factor de calibración (dB, default: 50)")
    args = parser.parse_args()
    process_with_pandas(args.input, args.output, args.flux_cal)

if __name__ == "__main__":
    main()
