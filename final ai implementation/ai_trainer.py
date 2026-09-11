import os
import sys
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from datetime import datetime, timezone
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt

SAMPLING_RATE = 100

def extract_features_from_axes(ax, ay, az):
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    n = len(mag)

    duration = n / SAMPLING_RATE
    peak_mag = np.max(np.abs(mag))
    rms_mag = np.sqrt(np.mean(mag**2))

    nsta, nlta = int(0.1 * SAMPLING_RATE), int(2.0 * SAMPLING_RATE)
    if nsta < nlta and nlta < n:
        csum = np.cumsum(mag**2)
        sta = (csum[nsta:] - csum[:-nsta]) / nsta
        lta = (csum[nlta:] - csum[:-nlta]) / nlta
        lta = np.where(lta == 0, 1e-10, lta)
        ratio = np.mean(sta[nlta - nsta:] / lta)
    else:
        ratio = 0.0

    yf = np.abs(rfft(mag))
    xf = rfftfreq(n, 1 / SAMPLING_RATE)
    dom_freq = float(xf[np.argmax(yf)]) if len(xf) > 0 else 0.0

    peak_ax = float(np.max(np.abs(ax)))
    peak_ay = float(np.max(np.abs(ay)))
    peak_az = float(np.max(np.abs(az)))
    rms_ax = float(np.sqrt(np.mean(ax**2)))
    rms_ay = float(np.sqrt(np.mean(ay**2)))
    rms_az = float(np.sqrt(np.mean(az**2)))

    return [duration, peak_mag, rms_mag, ratio, dom_freq,
            peak_ax, peak_ay, peak_az, rms_ax, rms_ay, rms_az]

def load_all_csv_features(data_dir=None):
    X = []
    y = []

    possible_dirs = [
        data_dir,
        "./converted_data",
        "../converted_data",
        os.path.join(os.path.dirname(__file__), "..", "converted_data"),
        os.path.join(os.path.dirname(__file__), "converted_data")
    ]
    
    target_dir = None
    labels_file = None
    for d in possible_dirs:
        if d and os.path.exists(os.path.join(d, "labels.csv")):
            target_dir = d
            labels_file = os.path.join(d, "labels.csv")
            break

    if not labels_file or not os.path.exists(labels_file):
        print(f"❌ Error: labels.csv not found in converted_data directory.")
        return np.array([]), np.array([])

    print(f"Loading datasets mapped in '{labels_file}'...")
    df_labels = pd.read_csv(labels_file)

    for idx, row in df_labels.iterrows():
        fname = row["filename"]
        label = row["label"]
        fpath = os.path.join(target_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue

        if all(c in df.columns for c in ("ax", "ay", "az")):
            ax = df["ax"].values
            ay = df["ay"].values
            az = df["az"].values
        elif all(c in df.columns for c in ("x", "y", "z")):
            ax = df["x"].values
            ay = df["y"].values
            az = df["z"].values
        else:
            continue

        feats = extract_features_from_axes(ax, ay, az)
        X.append(feats)
        y.append(label)

    return np.array(X), np.array(y)

def get_next_version_num():
    v = 1
    while os.path.exists(f"earthquake_ai_model_v{v}.pkl"):
        v += 1
    return v

if __name__ == "__main__":
    print("Loading STEAD CSV files and extracting 3-axis features...")
    X, y = load_all_csv_features()
    if len(X) == 0:
        print("No CSV files found via labels.csv. Run stead_to_csv.py first.")
        raise SystemExit

    feature_names = ["duration", "peak_mag", "rms_mag", "sta_lta_ratio", "dom_freq",
                     "peak_ax", "peak_ay", "peak_az", "rms_ax", "rms_ay", "rms_az"]

    df_features = pd.DataFrame(X, columns=feature_names)
    df_features["label"] = y
    df_features.to_csv("training_features.csv", index=False)
    print("Saved training_features.csv")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, min_samples_leaf=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy on test set: {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # Feature importance plot
    importances = clf.feature_importances_
    plt.figure(figsize=(12, 6))
    bars = plt.bar(feature_names, importances, color="#0284c7", edgecolor="#0369a1")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.title("Random Forest Feature Importances (11 STEAD Features)", fontsize=12, fontweight='bold')
    plt.ylabel("Importance Weight (Gini Impurity Reduction)", fontsize=10)
    
    max_imp = max(importances) if len(importances) > 0 else 0.2
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.003,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e293b')
    plt.ylim(0, max_imp * 1.18)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plot_path1 = "feature_importance.png"
    plot_path2 = os.path.join(os.path.dirname(__file__), "..", "feature_importance.png")
    plt.savefig(plot_path1, dpi=200)
    plt.savefig(plot_path2, dpi=200)
    print(f"Saved feature importance plot: {plot_path1}")
    plt.close()

    # Model Versioning & Manifest
    version_num = get_next_version_num()
    versioned_filename = f"earthquake_ai_model_v{version_num}.pkl"
    joblib.dump(clf, versioned_filename)
    joblib.dump(clf, "earthquake_ai_model.pkl")

    # Save to root workspace dir as well
    root_model_path = os.path.join(os.path.dirname(__file__), "..", "earthquake_ai_model.pkl")
    joblib.dump(clf, root_model_path)

    manifest = {
        "active_version": f"v{version_num}",
        "versioned_filename": versioned_filename,
        "default_filename": "earthquake_ai_model.pkl",
        "accuracy": round(float(acc), 4),
        "test_samples": len(y_test),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names
    }

    with open("model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    root_manifest = os.path.join(os.path.dirname(__file__), "..", "model_manifest.json")
    with open(root_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Model version saved: {versioned_filename} (and active earthquake_ai_model.pkl)")
    print(f"✅ Saved model_manifest.json (Active version: v{version_num}, Accuracy: {acc*100:.2f}%)")
