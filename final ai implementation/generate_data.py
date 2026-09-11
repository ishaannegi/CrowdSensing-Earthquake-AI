"""
SYNTHETIC SENSOR DATA GENERATOR (DEMO / FALLBACK MODE ONLY)
===========================================================
NOTE: The primary AI earthquake model in this repository is trained on 
real STEAD (Stanford Earthquake Dataset) data stored in ./converted_data/ 
mapped by labels.csv.

This script is maintained as an optional fallback / synthetic data 
generator for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd
import os

# total files to generate
REAL_FILES = 20
FAKE_FILES = 20
SAMPLES = 6000
SAMPLING_RATE = 100

# output folder (current directory)
OUT_DIR = '.'

np.random.seed(42)

def generate_real_quake(idx):
    t = np.arange(SAMPLES) / SAMPLING_RATE

    # base noise for each axis
    ax = np.random.randn(SAMPLES) * np.random.uniform(0.1, 0.6)
    ay = np.random.randn(SAMPLES) * np.random.uniform(0.1, 0.6)
    az = np.random.randn(SAMPLES) * np.random.uniform(0.1, 0.6)

    # slow drift
    drift_ax = np.sin(np.linspace(0, np.pi * np.random.uniform(0.5, 2.0), SAMPLES)) * np.random.uniform(0.02, 0.15)
    drift_ay = np.sin(np.linspace(0, np.pi * np.random.uniform(0.5, 2.0), SAMPLES)) * np.random.uniform(0.02, 0.15)
    drift_az = np.sin(np.linspace(0, np.pi * np.random.uniform(0.5, 2.0), SAMPLES)) * np.random.uniform(0.02, 0.15)

    ax += drift_ax
    ay += drift_ay
    az += drift_az

    # quake event
    quake_strength = np.random.uniform(2.0, 6.0)
    start = np.random.randint(1500, 3500)
    length = np.random.randint(200, 700)

    envelope = np.hanning(length) * quake_strength

    # AX: no shift
    ax[start:start+length] += envelope * np.random.uniform(0.8, 1.2)

    # AY: shifted, but safe
    shift_y = np.random.randint(-30, 30)
    y_start = start + shift_y
    y_start = max(0, min(y_start, SAMPLES - length))
    ay[y_start:y_start+length] += envelope * np.random.uniform(0.6, 1.1)

    # AZ: shifted, but safe
    shift_z = np.random.randint(-50, 50)
    z_start = start + shift_z
    z_start = max(0, min(z_start, SAMPLES - length))
    az[z_start:z_start+length] += envelope * np.random.uniform(0.7, 1.3)

    # aftershock (safe)
    if np.random.rand() < 0.3:
        after = start + length + np.random.randint(300, 800)
        after = min(after, SAMPLES - 200)
        alen = np.random.randint(80, 200)
        env2 = np.hanning(alen) * quake_strength * np.random.uniform(0.2, 0.6)

        ax[after:after+alen] += env2 * np.random.uniform(0.5, 1.0)
        ay[after:after+alen] += env2 * np.random.uniform(0.4, 0.9)
        az[after:after+alen] += env2 * np.random.uniform(0.5, 1.1)

    df = pd.DataFrame({"ax": ax, "ay": ay, "az": az})
    fname = os.path.join(OUT_DIR, f"sensor_data_real_{idx}.csv")
    df.to_csv(fname, index=False)
    print(f"✅ Created {fname}")


def generate_fake_vibration(idx):
    t = np.arange(SAMPLES) / SAMPLING_RATE

    ax = np.random.randn(SAMPLES) * np.random.uniform(0.1, 0.8)
    ay = np.random.randn(SAMPLES) * np.random.uniform(0.1, 0.8)
    az = np.random.randn(SAMPLES) * np.random.uniform(0.1, 0.8)

    # hum-like vibrations (traffic / machinery)
    for freq_scale, axis in [
        (np.random.uniform(0.5, 5.0), "ax"),
        (np.random.uniform(0.5, 5.0), "ay"),
        (np.random.uniform(0.5, 5.0), "az"),
    ]:
        hum = np.sin(2 * np.pi * np.linspace(0, freq_scale * 5, SAMPLES)) * np.random.uniform(0.05, 0.6)

        if axis == "ax":
            ax += hum
        elif axis == "ay":
            ay += hum * np.random.uniform(0.6, 1.0)
        else:
            az += hum * np.random.uniform(0.4, 0.9)

    df = pd.DataFrame({"ax": ax, "ay": ay, "az": az})
    fname = os.path.join(OUT_DIR, f"sensor_data_fake_{idx}.csv")
    df.to_csv(fname, index=False)
    print(f"✅ Created {fname}")


def introduce_label_errors():
    """Randomly swap some real/fake labels to simulate imperfect data."""
    all_files = [f for f in os.listdir(OUT_DIR) if f.startswith("sensor_data_") and f.endswith(".csv")]
    swap_count = max(1, len(all_files) // 15)
    np.random.shuffle(all_files)
    swapped = all_files[:swap_count]

    for f in swapped:
        old_name = os.path.join(OUT_DIR, f)
        if "real" in f:
            new_name = old_name.replace("real", "fake_error")
        else:
            new_name = old_name.replace("fake", "real_error")
        os.rename(old_name, new_name)
        print(f"⚠️ Introduced label error: {os.path.basename(new_name)}")


def main():
    print("Generating realistic 3-axis sensor data...\n")

    # remove old data
    for f in os.listdir(OUT_DIR):
        if f.startswith("sensor_data_") and f.endswith(".csv"):
            os.remove(os.path.join(OUT_DIR, f))

    # generate new
    for i in range(REAL_FILES):
        generate_real_quake(i)
    for i in range(FAKE_FILES):
        generate_fake_vibration(i)

    introduce_label_errors()

    print("\n✅ File generation complete.")
    print(f"Generated {REAL_FILES + FAKE_FILES} files.")


if __name__ == "__main__":
    main()
