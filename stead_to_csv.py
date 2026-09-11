import argparse
import os
import sys
import numpy as np
import pandas as pd
import h5py

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

COLUMN_NAMES = ["ax", "ay", "az"]
SAMPLE_RATE_HZ = 100  # STEAD default


def inspect_hdf5(filepath):
    print(f"==================================================")
    print(f"[INSPECT] HDF5 STRUCTURE INSPECTION: {filepath}")
    print(f"==================================================")

    if not os.path.exists(filepath):
        print(f"[ERROR] Error: File '{filepath}' not found.")
        sys.exit(1)

    with h5py.File(filepath, "r") as f:
        print(f"Top-level keys: {list(f.keys())}\n")

        if "traces" in f and isinstance(f["traces"], h5py.Dataset):
            ds = f["traces"]
            print(f"Found 'traces' dataset:")
            print(f"  - Full Shape: {ds.shape}")
            print(f"  - Data Type: {ds.dtype}")
            print(f"  - Total Traces (samples): {ds.shape[0]}")
            print(f"  - Data points per trace: {ds.shape[1]}")
            if ds.ndim > 2:
                print(f"  - Channels per trace: {ds.shape[2]} (ax, ay, az)")
            print(f"\nSample trace 0 shape: {ds[0].shape}")
            print(f"Sample trace 0 head (first 5 timesteps):\n{ds[0][:5]}\n")
        else:
            # Check for group-based traces
            grp = f.get("data", f)
            keys = [k for k in grp.keys() if k not in ["metadata", "p_arrival", "s_arrival"]]
            print(f"Found {len(keys)} trace entries under group.")
            if keys:
                sample_key = keys[0]
                ds = grp[sample_key]
                print(f"Sample trace key: '{sample_key}'")
                print(f"  - Shape: {ds.shape}, Dtype: {ds.dtype}")
                print(f"  - Attrs: {dict(ds.attrs)}")

        for k in f.keys():
            if k != "traces":
                item = f[k]
                if isinstance(item, h5py.Dataset):
                    print(f"Metadata dataset '{k}': shape={item.shape}, dtype={item.dtype}")


def convert_hdf5(hdf5_path, label, n_samples, out_prefix, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    records = []

    print(f"==================================================")
    print(f"[CONVERT] CONVERTING: {hdf5_path}")
    print(f"   - Target Label: {label}")
    print(f"   - Number of Samples: {n_samples}")
    print(f"   - Output Directory: {out_dir}")
    print(f"==================================================")

    if not os.path.exists(hdf5_path):
        print(f"[ERROR] Error: File '{hdf5_path}' not found.")
        sys.exit(1)

    with h5py.File(hdf5_path, "r") as f:
        if "traces" in f and isinstance(f["traces"], h5py.Dataset):
            traces = f["traces"]
            total_available = traces.shape[0]
            count = min(n_samples, total_available)
            print(f"Extracting {count} traces from total {total_available} available in 'traces' dataset...")

            for i in range(count):
                waveform = np.array(traces[i])  # shape (6000, 3)
                if waveform.shape[0] == 3 and waveform.shape[1] != 3:
                    waveform = waveform.T

                df = pd.DataFrame(waveform, columns=COLUMN_NAMES)
                filename = f"{out_prefix}_{i}.csv"
                out_path = os.path.join(out_dir, filename)
                df.to_csv(out_path, index=False)

                records.append({"filename": filename, "label": label})

                if (i + 1) % 25 == 0 or (i + 1) == count:
                    print(f"  - Converted {i + 1}/{count} files...")
        else:
            grp = f.get("data", f)
            keys = [k for k in grp.keys() if k not in ["metadata", "p_arrival", "s_arrival"]]
            if not keys:
                print("[ERROR] Error: No trace datasets or groups found in HDF5.")
                sys.exit(1)

            count = min(n_samples, len(keys))
            print(f"Extracting {count} traces from group...")

            for i, key in enumerate(keys[:count]):
                waveform = np.array(grp[key])
                if waveform.shape[0] == 3 and waveform.shape[1] != 3:
                    waveform = waveform.T

                df = pd.DataFrame(waveform, columns=COLUMN_NAMES)
                filename = f"{out_prefix}_{i}.csv"
                out_path = os.path.join(out_dir, filename)
                df.to_csv(out_path, index=False)

                records.append({"filename": filename, "label": label})

                if (i + 1) % 25 == 0 or (i + 1) == count:
                    print(f"  - Converted {i + 1}/{count} files...")

    print(f"[SUCCESS] Converted {len(records)} files for label '{label}'.\n")
    return records


def save_labels_csv(records, out_dir):
    labels_path = os.path.join(out_dir, "labels.csv")
    if os.path.exists(labels_path):
        df_existing = pd.read_csv(labels_path)
    else:
        df_existing = pd.DataFrame(columns=["filename", "label"])

    df_new = pd.DataFrame(records)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=["filename"], keep="last")
    df_combined.to_csv(labels_path, index=False)
    print(f"[SUCCESS] Created/Updated labels file: '{labels_path}' (Total mapped files: {len(df_combined)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert STEAD HDF5 datasets to CSV format")
    p.add_argument("--hdf5", help="Path to test.hdf5 or test_noise.hdf5")
    p.add_argument("--inspect", help="Inspect structure of given hdf5 file")
    p.add_argument("--label", choices=["real_quake", "fake_vibration"], help="Label for this batch")
    p.add_argument("--n", type=int, default=150, help="Number of samples to convert (default: 150)")
    p.add_argument("--out", help="Output filename prefix (default: uses label name)")
    p.add_argument("--out-dir", default="./converted_data", help="Output directory")

    args = p.parse_args()

    if args.inspect:
        inspect_hdf5(args.inspect)
    elif args.hdf5 and args.label:
        prefix = args.out if args.out else args.label
        recs = convert_hdf5(args.hdf5, args.label, args.n, prefix, args.out_dir)
        save_labels_csv(recs, args.out_dir)
    else:
        p.print_help()