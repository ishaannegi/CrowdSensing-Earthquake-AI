import numpy as np
import matplotlib.pyplot as plt

print("Enter the following parameters for STA/LTA Detector:\n")

SAMPLING_RATE = int(input("Sampling rate (samples/sec): "))
T_short = float(input("Short-term window (seconds): "))
T_long = float(input("Long-term window (seconds): "))
TRIGGER_ON = float(input("Trigger ON ratio: "))
TRIGGER_OFF = float(input("Trigger OFF ratio: "))
TOTAL_SAMPLES = int(input("Total samples to simulate: "))
P_WAVE_START = int(input("P-wave start index: "))
P_WAVE_END = int(input("P-wave end index: "))

np.random.seed(42)
seismic_noise = np.random.randn(TOTAL_SAMPLES) * 0.5
seismic_signal = seismic_noise.copy()
seismic_signal[P_WAVE_START:P_WAVE_END] += 5.0

def run_sta_lta_detector(signal, sampling_rate, t_short, t_long, trig_on, trig_off):
    nsta = int(t_short * sampling_rate)
    nlta = int(t_long * sampling_rate)
    squared_signal = signal ** 2
    csta = np.cumsum(squared_signal)
    sta = (csta[nsta:] - csta[:-nsta]) / nsta
    clta = np.cumsum(squared_signal)
    lta = (clta[nlta:] - clta[:-nlta]) / nlta
    lta = np.where(lta == 0, 1e-10, lta)
    ratio = sta[nlta - nsta:] / lta
    on, off = [], []
    triggered = False
    for i, val in enumerate(ratio):
        if not triggered and val > trig_on:
            on.append(i + nlta)
            triggered = True
        elif triggered and val < trig_off:
            off.append(i + nlta)
            triggered = False
    if triggered and len(on) > len(off):
        off.append(len(signal) - 1)
    return on, off, ratio

onsets, offsets, ratio = run_sta_lta_detector(
    seismic_signal, SAMPLING_RATE, T_short, T_long, TRIGGER_ON, TRIGGER_OFF
)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(seismic_signal, label="Seismic Signal")
for o in onsets:
    plt.axvline(o, color='r', linestyle='--', linewidth=1, label='Event Onset' if o == onsets[0] else "")
for f in offsets:
    plt.axvline(f, color='g', linestyle='--', linewidth=1, label='Event Offset' if f == offsets[0] else "")
plt.title("Seismic Signal with Detected Events", fontsize=14)
plt.ylabel("Amplitude", fontsize=12)
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
time_axis = np.arange(len(ratio)) + int(T_long * SAMPLING_RATE)
plt.plot(time_axis, ratio, label="STA/LTA Ratio", color='k')
plt.axhline(TRIGGER_ON, color='r', linestyle='-', linewidth=1, label=f'Trigger ON ({TRIGGER_ON})')
plt.axhline(TRIGGER_OFF, color='g', linestyle='-', linewidth=1, label=f'Trigger OFF ({TRIGGER_OFF})')
plt.title("STA/LTA Ratio", fontsize=14)
plt.xlabel("Sample Index", fontsize=12)
plt.ylabel("Ratio", fontsize=12)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("\n--- Detection Results ---")
print(f"P-wave was at indices: {P_WAVE_START} to {P_WAVE_END}")
print(f"Event start indices (Onsets): {onsets}")
print(f"Event end indices (Offsets): {offsets}")
print("-------------------------")
