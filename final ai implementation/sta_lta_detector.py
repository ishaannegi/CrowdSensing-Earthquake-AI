import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

SAMPLING_RATE = 100
T_SHORT = 0.1
T_LONG = 2.0
TRIG_ON = 10.0
TRIG_OFF = 2.0

def run_sta_lta_detector(signal, sampling_rate=SAMPLING_RATE, t_short=T_SHORT, t_long=T_LONG, trig_on=TRIG_ON, trig_off=TRIG_OFF):
    nsta = int(t_short * sampling_rate)
    nlta = int(t_long * sampling_rate)
    if nsta >= nlta or nlta >= len(signal):
        return [], [], np.array([])
    squared = signal**2
    csum = np.cumsum(squared)
    sta = (csum[nsta:] - csum[:-nsta]) / nsta
    lta = (csum[nlta:] - csum[:-nlta]) / nlta
    lta = np.where(lta == 0, 1e-10, lta)
    ratio = sta[nlta - nsta:] / lta
    on, off = [], []
    trig = False
    for i, val in enumerate(ratio):
        if not trig and val > trig_on:
            on.append(i + nlta)
            trig = True
        elif trig and val < trig_off:
            off.append(i + nlta)
            trig = False
    if trig and len(on) > len(off):
        off.append(len(signal)-1)
    return on, off, ratio

def process_file(filename, color):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("File not found:", filename)
        return
    if not all(c in df.columns for c in ("ax","ay","az")):
        print("Missing columns in", filename)
        return
    ax = df["ax"].values
    ay = df["ay"].values
    az = df["az"].values
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    on, off, ratio = run_sta_lta_detector(mag)
    if on and off and on[0] < off[0]:
        start, end = on[0], off[0]
        duration = (end - start) / SAMPLING_RATE
        peak = np.max(np.abs(mag[start:end]))
    else:
        duration, peak = 0, 0
    print(f"\n{filename}: Duration={duration:.2f}s Peak={peak:.2f}")
    plt.plot(mag, label=f"{os.path.basename(filename)} (peak {peak:.2f})", color=color, alpha=0.8)
    for o in on:
        plt.axvline(o, color='r', linestyle='--', linewidth=1)
    for f in off:
        plt.axvline(f, color='g', linestyle='--', linewidth=1)

if __name__ == "__main__":
    files = sorted([f for f in os.listdir('.') if f.startswith("sensor_data_") and f.endswith(".csv")])
    if len(files) == 0:
        print("No sensor_data_*.csv files found. Run generate_data.py first.")
        sys.exit()
    plt.figure(figsize=(14,7))
    colors = ['blue','orange','green','purple','brown','cyan']
    for i, fname in enumerate(files[:6]):
        process_file(fname, colors[i % len(colors)])
    plt.title("STA/LTA - 3 axis magnitude")
    plt.xlabel("Sample index")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
