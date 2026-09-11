import numpy as np
import pandas as pd
import json
import socket
import joblib
from scipy.fft import rfft, rfftfreq
from datetime import datetime, timezone

SAMPLING_RATE = 100
MODEL_PATH = "earthquake_ai_model.pkl"

def extract_features(ax, ay, az):
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    n = len(mag)
    duration = n / SAMPLING_RATE
    peak_mag = float(np.max(np.abs(mag)))
    rms_mag = float(np.sqrt(np.mean(mag**2)))

    # STA/LTA on magnitude
    nsta, nlta = int(0.1 * SAMPLING_RATE), int(2.0 * SAMPLING_RATE)
    if nsta < nlta and nlta < n:
        csum = np.cumsum(mag**2)
        sta = (csum[nsta:] - csum[:-nsta]) / nsta
        lta = (csum[nlta:] - csum[:-nlta]) / nlta
        lta = np.where(lta == 0, 1e-10, lta)
        ratio = float(np.mean(sta[nlta - nsta:] / lta))
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

def run_sta_lta_detector(signal, sampling_rate=SAMPLING_RATE, t_short=0.1, t_long=2.0, trig_on=10.0, trig_off=2.0):
    nsta = int(t_short * sampling_rate)
    nlta = int(t_long * sampling_rate)
    if nsta >= nlta or nlta >= len(signal):
        return [], [], np.array([])
    sq = signal**2
    csum = np.cumsum(sq)
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

def send_to_server(payload, server_ip="127.0.0.1", port=9000):
    raw = json.dumps(payload).encode("utf-8")
    try:
        with socket.create_connection((server_ip, port), timeout=5) as s:
            s.sendall(raw)
            s.shutdown(socket.SHUT_WR)
            reply = s.recv(256).decode("utf-8", errors="ignore")
            print("[SERVER REPLY]", reply)
            return True
    except Exception as e:
        print("[CLIENT ERROR] Send failed:", e)
        return False

def main():
    import sys
    import argparse

    parser.add_argument("filename", nargs="?", default=None, help="Path to CSV sensor data file")
    parser.add_argument("--send", action="store_true", help="Automatically send result to server")
    parser.add_argument("--ip", default="127.0.0.1", help="Server IP address")
    parser.add_argument("--port", type=int, default=9000, help="Server port number")
    parser.add_argument("--node-id", default="Node-1 (Delhi CP)", help="Node Identifier")
    parser.add_argument("--lat", type=float, default=28.6139, help="Latitude")
    parser.add_argument("--lon", type=float, default=77.2090, help="Longitude")
    args = parser.parse_args()

    try:
        model = joblib.load(MODEL_PATH)
        print("✅ AI model loaded.")
    except Exception as e:
        print("❌ Model load error:", e)
        return

    if args.filename:
        fname = args.filename
    else:
        fname = input("Enter CSV filename (ax,ay,az): ").strip()

    try:
        df = pd.read_csv(fname)
    except Exception as e:
        print("❌ File read error:", e)
        return

    if all(c in df.columns for c in ("ax", "ay", "az")):
        ax = df["ax"].values
        ay = df["ay"].values
        az = df["az"].values
    elif all(c in df.columns for c in ("x", "y", "z")):
        ax = df["x"].values
        ay = df["y"].values
        az = df["z"].values
    else:
        print("[ERROR] CSV must contain columns: ax, ay, az (or x, y, z)")
        return

    feat = extract_features(ax, ay, az)
    pred = model.predict([feat])[0]
    # proper confidence: probability of predicted class
    try:
        proba = model.predict_proba([feat])[0]
        conf = round(float(proba[list(model.classes_).index(pred)]), 4)
    except Exception:
        conf = 0.0

    # STA/LTA on magnitude for display
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    on, off, ratio = run_sta_lta_detector(mag)

    from datetime import timezone
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "node_id": args.node_id,
        "location": {"lat": args.lat, "lon": args.lon},
        "file": fname,
        "prediction": str(pred),
        "confidence": conf,
        "features": {
            "duration": feat[0],
            "peak_mag": feat[1],
            "rms_mag": feat[2],
            "sta_lta_ratio": feat[3],
            "dom_freq": feat[4],
            "peak_ax": feat[5],
            "peak_ay": feat[6],
            "peak_az": feat[7]
        }
    }

    print("\n=== Detection Result ===")
    print("Prediction:", pred)
    print("Confidence:", conf)
    print("Peak (mag):", feat[1])

    if args.filename:
        if args.send:
            send_to_server(payload, args.ip, args.port)
        else:
            print("Not sent.")
    else:
        send = input("Send result to server? (y/n): ").strip().lower()
        if send == "y":
            ip = input("Server IP [127.0.0.1]: ").strip() or "127.0.0.1"
            send_to_server(payload, ip, 9000)
        else:
            print("Not sent.")

if __name__ == "__main__":
    main()
