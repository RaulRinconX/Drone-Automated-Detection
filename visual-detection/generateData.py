import subprocess

print("Executing hackrf_sweep...")
subprocess.run([
    "hackrf_sweep",
    "-f", "2350:2550",
    "-a", "1",
    "-p", "1",
    "-l", "40",
    "-g", "62",
    "-w", "20000",
    "-N", "2500",
    "-r", "prueba.csv"
], check=True, shell=False, timeout=None)

print("Processing deteccion.csv with tratamiento_datos_pandas.py...")
subprocess.run([
    "python3", "../tratamiento_datos_pandas.py",
    "-i", "prueba.csv",
    "-o", "prueba_tratada.csv",
    "-f", "50"

], check=True, shell=False)
