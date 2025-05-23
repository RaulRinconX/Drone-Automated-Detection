#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import math

def round_to_sigfigs(num, sigfigs):
    """Redondea num a sigfigs cifras significativas."""
    if num == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(num)))
    factor = 10 ** (sigfigs - 1 - magnitude)
    return round(num * factor) / factor

def process_with_pandas(input_csv: str, output_csv: str, flux_cal_db: float):
    # 1) Leer CSV raw, saltando la primera línea
    df = pd.read_csv(input_csv, header=None, skiprows=1)

    # 2) Nombrar columnas: 6 fijas + el resto db1, db2, ...
    ncols = df.shape[1]
    base = ['date', 'time', 'hz_low', 'hz_high', 'hz_bin_width', 'num_samples']
    db_cols = [f'db{i+1}' for i in range(ncols - len(base))]
    df.columns = base + db_cols

    # 3) Timestamp combinado (sin decimales de segundo)
    df['Timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'].str.split('.').str[0],
        format='%Y-%m-%d %H:%M:%S'
    ).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 4) Convertir dB a mW/Hz
    lin_mw_per_hz = 10 ** (df[db_cols] / 10.0)

    # 5) Filtrar sólo bins > -95 dBm (Java usa ese umbral)
    mask = df[db_cols] > -95.0
    lin_mw_filtered = lin_mw_per_hz.where(mask, other=0.0)

    # 6) Multiplicar por ancho de bin para obtener mW/bin
    lin_mw_per_bin = lin_mw_filtered.mul(df['hz_bin_width'], axis=0)

    # 7) Sumar potencia total en mW
    power_sum_mw = lin_mw_per_bin.sum(axis=1)

    # 8) Encontrar Max Amplitude y la posición del bin
    df['Max Amplitude [dBm]'] = df[db_cols].max(axis=1)
    idx_max = df[db_cols].values.argmax(axis=1)

    # 9) Calcular Frequency [MHz] con redondeo al paso de bin
    #    freqStep = hz_bin_width (Hz) / 1e6 -> MHz
    freq_step_mhz = df['hz_bin_width'] / 1e6
    #    freq_unrounded = hz_low/1e6 + freq_step * idx_max
    freq_unrounded = df['hz_low'] / 1e6 + freq_step_mhz * idx_max
    steps = (1.0 / freq_step_mhz).round()  # número de pasos por MHz
    df['Frequency [MHz]'] = ( (steps * freq_unrounded).round() / steps )

    # 10) Calibración lineal desde dB
    calibration_lin = 10 ** (flux_cal_db / 10.0)

    # 11) Factor electromagnético: 4π·(f/1e3)²·1e18 / c²
    c = 299_792_458  # m/s
    f_ghz = df['Frequency [MHz]'] / 1e3  # GHz
    em_factor = (4 * math.pi * f_ghz**2 * 1e18) / (c**2)

    # 12) Power Flux Density [µW/m²]
    df['Power Flux Density [µW/m²]'] = power_sum_mw * calibration_lin * em_factor

    # 13) Total Spectrum Power [dBm]: 10·log10(power_sum_mw)
    df['Total Spectrum Power [dBm]'] = 10 * np.log10(power_sum_mw.replace(0, np.nan))
    df['Total Spectrum Power [dBm]'] = df['Total Spectrum Power [dBm]'].fillna(-150.0)

    # 14) Redondear y formatear:
    df['Total Spectrum Power [dBm]']    = df['Total Spectrum Power [dBm]'].map(lambda x: f"{x:.1f}")
    df['Max Amplitude [dBm]']           = df['Max Amplitude [dBm]'].map(lambda x: f"{x:.1f}")
    df['Frequency [MHz]']               = df['Frequency [MHz]'].map(lambda x: f"{x:.2f}")
    # Power Flux a 2 cifras significativas
    df['Power Flux Density [µW/m²]']    = df['Power Flux Density [µW/m²]'] \
        .map(lambda x: f"{round_to_sigfigs(x, 2):.2g}")

    # 15) Exportar columnas finales
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
    parser.add_argument('-i', '--input',  required=True, help="CSV raw, se ignora la primera línea")
    parser.add_argument('-o', '--output', required=True, help="CSV de salida")
    parser.add_argument('-f', '--flux-cal', type=float, default=50.0,
                        help="Factor de calibración en dB (default: 50)")
    args = parser.parse_args()

    process_with_pandas(args.input, args.output, args.flux_cal)

if __name__ == "__main__":
    main()
