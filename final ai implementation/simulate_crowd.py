import os
import sys
import time
import json
import socket
import logging
import threading
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from scipy.fft import rfft, rfftfreq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CrowdSimulator")

SAMPLING_RATE = 100
SERVER_IP = "127.0.0.1"
SERVER_PORT = 9000

NODES = [
    {"node_id": "Node-1 (Delhi CP)", "lat": 28.6139, "lon": 77.2090},
    {"node_id": "Node-2 (Noida Sec 62)", "lat": 28.6280, "lon": 77.3649},
    {"node_id": "Node-3 (Gurugram CyberCity)", "lat": 28.4595, "lon": 77.0266},
    {"node_id": "Node-4 (Faridabad Central)", "lat": 28.4089, "lon": 77.3178},
    {"node_id": "Node-5 (Ghaziabad)", "lat": 28.6692, "lon": 77.4538},
    {"node_id": "Node-6 (Delhi Dwarka)", "lat": 28.5921, "lon": 77.0460},
    {"node_id": "Node-7 (Greater Noida)", "lat": 28.4744, "lon": 77.5040},
    {"node_id": "Node-8 (Delhi Rohini)", "lat": 28.7041, "lon": 77.1025},
    {"node_id": "Node-9 (CP East)", "lat": 28.6200, "lon": 77.2200},
    {"node_id": "Node-10 (Karol Bagh)", "lat": 28.6500, "lon": 77.1900},
]

def extract_features(ax, ay, az):
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    n = len(mag)
    duration = n / SAMPLING_RATE
    peak_mag = float(np.max(np.abs(mag)))
    rms_mag = float(np.sqrt(np.mean(mag**2)))

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

def process_and_send(node_info, csv_file, scenario=None):
    if not os.path.exists(csv_file):
        logger.error(f"File not found: {csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file)
        if all(c in df.columns for c in ("ax", "ay", "az")):
            ax, ay, az = df["ax"].values, df["ay"].values, df["az"].values
        elif all(c in df.columns for c in ("x", "y", "z")):
            ax, ay, az = df["x"].values, df["y"].values, df["z"].values
        else:
            return None
    except Exception as e:
        logger.error(f"Read error for {csv_file}: {e}")
        return None

    feat = extract_features(ax, ay, az)

    payload = {
        "node_id": node_info["node_id"],
        "location": {"lat": node_info["lat"], "lon": node_info["lon"]},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "file": os.path.basename(csv_file),
        "scenario": scenario,
        "features": {
            "duration": feat[0],
            "peak_mag": feat[1],
            "rms_mag": feat[2],
            "sta_lta_ratio": feat[3],
            "dom_freq": feat[4],
            "peak_ax": feat[5],
            "peak_ay": feat[6],
            "peak_az": feat[7],
            "rms_ax": feat[8],
            "rms_ay": feat[9],
            "rms_az": feat[10]
        }
    }

    raw = json.dumps(payload).encode("utf-8")
    try:
        with socket.create_connection((SERVER_IP, SERVER_PORT), timeout=5) as s:
            s.sendall(raw)
            reply = s.recv(4096).decode("utf-8", errors="ignore")
            resp_data = json.loads(reply)
            return resp_data
    except (socket.timeout, socket.error, ConnectionResetError, BrokenPipeError) as e:
        logger.warning(f"Transmission error from {node_info['node_id']}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected transmission error for {node_info['node_id']}: {e}")
        return None

def send_batch_concurrently(node_files, scenario):
    results = [None] * len(node_files)
    threads = []

    def worker(idx, node_info, csv_file):
        res = process_and_send(node_info, csv_file, scenario=scenario)
        results[idx] = res

    logger.info(f"⚡ [CONCURRENT SPAWN] Launching {len(node_files)} simultaneous client threads for Scenario {scenario}...")
    for idx, (node_info, csv_file) in enumerate(node_files):
        t = threading.Thread(target=worker, args=(idx, node_info, csv_file), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=5.0)

    for res in results:
        if res:
            logger.info(f"  [CLIENT RESPONSE] Node: {res.get('node_id')} -> Alert: {res.get('alert_level')}")

def run_scenario_1(gpath):
    logger.info("--------------------------------------------------")
    logger.info("SCENARIO 1: Single Node Local Vibration (False Alarm)")
    logger.info("  -> Only Node-1 detects earthquake; other nodes send noise concurrently.")
    logger.info("--------------------------------------------------")
    node_files = [(node, gpath("real_quake_0.csv") if i == 0 else gpath(f"fake_vibration_{i}.csv")) for i, node in enumerate(NODES[:8])]
    send_batch_concurrently(node_files, scenario=1)
    logger.info("  ⏳ [PAUSE] Waiting 5 seconds to observe live dashboard state (LOCAL VIBRATION)...")
    time.sleep(5.0)

def run_scenario_2(gpath):
    logger.info("--------------------------------------------------")
    logger.info("SCENARIO 2: Local Cluster Consensus (15km Radius Confirmed Alert)")
    logger.info("  -> Node-1 (Delhi CP), Node-9 (CP East), Node-10 (Karol Bagh) within <5km radius detect quake!")
    logger.info("  -> Step-by-step presentation mode: 5s initial delay + 3s delay per node reading.")
    logger.info("--------------------------------------------------")
    logger.info("  ⏳ [STARTUP DELAY] Pausing 5 seconds before initiating Scenario 2 telemetry...")
    time.sleep(5.0)

    cluster_nodes = [NODES[0], NODES[8], NODES[9]]  # Node-1 (Delhi CP), Node-9 (CP East), Node-10 (Karol Bagh)
    node_files = [(node, gpath(f"real_quake_{i}.csv")) for i, node in enumerate(cluster_nodes)]

    for idx, (node_info, csv_file) in enumerate(node_files):
        logger.info(f"📡 [NODE {idx+1}/3] Transmitting seismic telemetry for {node_info['node_id']}...")
        res = process_and_send(node_info, csv_file, scenario=2)
        if res:
            mw_info = f" | Mw {res.get('magnitude_mw')}" if res.get('magnitude_mw') else ""
            logger.info(f"   [SERVER RESPONSE] Node: {res.get('node_id')} -> Alert: {res.get('alert_level')}{mw_info}")
        
        if idx < len(node_files) - 1:
            logger.info(f"   ⏳ [STEP DELAY] Pausing 3 seconds before next node reading...")
            time.sleep(3.0)

    logger.info("  ⏳ [OBSERVATION PAUSE] Pausing 5 seconds to observe live confirmed alert dashboard state (CONFIRMED ALERT)...")
    time.sleep(5.0)

def run_scenario_3(gpath):
    logger.info("--------------------------------------------------")
    logger.info("SCENARIO 3: Distant Nodes Test (>40km Apart Spatial Filtering)")
    logger.info("  -> Node-3 (Gurugram) and Node-7 (Greater Noida) are ~46.7 km apart (>15km radius).")
    logger.info("  -> Both detect earthquake concurrently, but FAIL spatial radius check -> stay LOCAL_VIBRATION!")
    logger.info("--------------------------------------------------")
    distant_nodes = [NODES[2], NODES[6]]  # Node-3 (Gurugram) and Node-7 (Greater Noida)
    node_files = [(node, gpath(f"real_quake_{i+3}.csv")) for i, node in enumerate(distant_nodes)]
    send_batch_concurrently(node_files, scenario=3)
    logger.info("  ⏳ [PAUSE] Waiting 5 seconds to observe live dashboard state (SPATIAL FILTERING)...")
    time.sleep(5.0)

def run_scenario_4(gpath):
    logger.info("--------------------------------------------------")
    logger.info("SCENARIO 4: Normal Ambient Noise Mode (System Reset)")
    logger.info("  -> All nodes send normal ambient vibration data concurrently.")
    logger.info("--------------------------------------------------")
    node_files = [(node, gpath(f"fake_vibration_{i}.csv")) for i, node in enumerate(NODES[:5])]
    send_batch_concurrently(node_files, scenario=4)
    logger.info("  ⏳ [PAUSE] Waiting 5 seconds to observe dashboard reset (SYSTEM NORMAL)...")
    time.sleep(5.0)

def run_test_scenarios(scenario_choice="all"):
    logger.info("==================================================")
    logger.info("   MULTI-NODE CROWDSENSING SIMULATOR (CONCURRENT)  ")
    logger.info("==================================================")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    converted_dir = os.path.join(script_dir, "..", "converted_data")

    def gpath(fname):
        return os.path.join(converted_dir, fname)

    sc_str = str(scenario_choice).lower().strip()

    if sc_str in ("1", "s1", "scenario1"):
        run_scenario_1(gpath)
    elif sc_str in ("2", "s2", "scenario2"):
        run_scenario_2(gpath)
    elif sc_str in ("3", "s3", "scenario3"):
        run_scenario_3(gpath)
    elif sc_str in ("4", "s4", "scenario4"):
        run_scenario_4(gpath)
    else:
        run_scenario_1(gpath)
        run_scenario_2(gpath)
        run_scenario_3(gpath)
        run_scenario_4(gpath)

    logger.info("🎉 SIMULATION SCENARIO EXECUTION FINISHED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrowdSensing Multi-Node Simulator")
    parser.add_argument("--scenario", "-s", type=str, default="all", help="Scenario number to run (1, 2, 3, 4, or all)")
    args = parser.parse_args()
    run_test_scenarios(args.scenario)
