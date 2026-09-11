import pandas as pd
import numpy as np
import sys

def generate_files():
    print("Generating sensor data files...")
    
    SAMPLING_RATE = 100
    np.random.seed(42)
    
    time = np.arange(0, 60, 1.0/SAMPLING_RATE)
    
    mag_real = np.random.normal(0, 0.5, len(time))
    real_start, real_end = 2500, 2600
    mag_real[real_start:real_end] += np.sin(np.linspace(0, 15*np.pi, real_end - real_start)) * 10
    
    try:
        pd.DataFrame({"time": time, "magnitude": mag_real}).to_csv("sensor_data_real.csv", index=False)
        print("- Created sensor_data_real.csv (real earthquake)")
    except Exception as e:
        print(f"ERROR creating sensor_data_real.csv: {e}", file=sys.stderr)
        return

    mag_fake = np.random.normal(0, 0.5, len(time))
    fake_start, fake_end = 1000, 1020
    mag_fake[fake_start:fake_end] += np.sin(np.linspace(0, 2*np.pi, fake_end - fake_start)) * 3
    
    try:
        pd.DataFrame({"time": time, "magnitude": mag_fake}).to_csv("sensor_data_fake.csv", index=False)
        print("- Created sensor_data_fake.csv (fake vibration)")
    except Exception as e:
        print(f"ERROR creating sensor_data_fake.csv: {e}", file=sys.stderr)
        return
        
    print("\nFile generation complete.")

if __name__ == "__main__":
    generate_files()