import os
import sys
import time
import json
import socket
import logging

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AdaptiveRadiusDemo")

SERVER_IP = "127.0.0.1"
SERVER_PORT = 9000

REAL_QUAKE_FEATS = [60.0, 64682.75, 11889.23, 1.077, 0.0, 31927.32, 59199.60, 54457.13, 6152.21, 7764.14, 6574.36]
SERVEREAL_QUAKE_FEATS = REAL_QUAKE_FEATS

def send_node_payload(node_id, lat, lon, peak_mag=18.5, is_reset=False, **kwargs):
    payload = {
        "node_id": node_id,
        "location": {"lat": lat, "lon": lon},
        "file": "real_quake_0.csv",
        "features": {
            "duration": REAL_QUAKE_FEATS[0] if not is_reset else 1.0,
            "peak_mag": REAL_QUAKE_FEATS[1] if not is_reset else 0.01,
            "rms_mag": REAL_QUAKE_FEATS[2] if not is_reset else 0.001,
            "sta_lta_ratio": REAL_QUAKE_FEATS[3] if not is_reset else 0.1,
            "dom_freq": REAL_QUAKE_FEATS[4] if not is_reset else 35.0,
            "peak_ax": REAL_QUAKE_FEATS[5] if not is_reset else 0.01,
            "peak_ay": REAL_QUAKE_FEATS[6] if not is_reset else 0.01,
            "peak_az": REAL_QUAKE_FEATS[7] if not is_reset else 0.01,
            "rms_ax": REAL_QUAKE_FEATS[8] if not is_reset else 0.001,
            "rms_ay": REAL_QUAKE_FEATS[9] if not is_reset else 0.001,
            "rms_az": REAL_QUAKE_FEATS[10] if not is_reset else 0.001
        }
    }
    try:
        raw = json.dumps(payload).encode("utf-8")
        with socket.create_connection((SERVER_IP, SERVER_PORT), timeout=5) as s:
            s.sendall(raw)
            reply = s.recv(4096).decode("utf-8", errors="ignore")
            return json.loads(reply)
    except Exception as e:
        logger.error(f"Error sending payload for {node_id}: {e}")
        return None

def run_adaptive_radius_demo():
    print("\n================================================================================")
    print(" 🔬 DEMONSTRATION: PROPOSAL 2 - DYNAMIC DENSITY-ADAPTIVE CONSENSUS RADIUS SCALING")
    print("================================================================================\n")
    
    # -------------------------------------------------------------------------
    # PHASE 1: Urban High-Density Cluster (4 Tightly Packed Downtown Nodes < 1.5km Spacing)
    # -------------------------------------------------------------------------
    print("📍 PHASE 1: URBAN HIGH-DENSITY CLUSTER (Delhi CP Downtown)")
    print("   Nodes are tightly packed (~1.1 km spacing). Expected Adaptive Radius: 5.0 km (Urban High Precision)\n")
    
    urban_nodes = [
        {"node_id": "UrbanNode-1 (Delhi CP)", "lat": 28.6139, "lon": 77.2090},
        {"node_id": "UrbanNode-2 (CP East)", "lat": 28.6200, "lon": 77.2200},
        {"node_id": "UrbanNode-3 (Karol Bagh)", "lat": 28.6250, "lon": 77.2100},
        {"node_id": "UrbanNode-4 (Connaught South)", "lat": 28.6100, "lon": 77.2150},
    ]

    for idx, node in enumerate(urban_nodes):
        print(f"   📡 Transmitting Telemetry for {node['node_id']}...")
        resp = send_node_payload(node["node_id"], node["lat"], node["lon"])
        if resp:
            print(f"      └── Server Response: Alert = {resp.get('alert_level')} | Mw = {resp.get('magnitude_mw', 'N/A')}")
        time.sleep(1.5)

    print("\n   ⏳ Pausing 5 seconds to observe 5.0 km CYAN consensus circle on web dashboard...")
    time.sleep(5.0)

    # -------------------------------------------------------------------------
    # PHASE 2: Standard Regional Cluster (Nodes Spaced ~6 km Apart)
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("📍 PHASE 2: STANDARD REGIONAL CLUSTER (Delhi CP to Noida/Ghaziabad)")
    print("   Nodes are moderately spaced (~6.2 km spacing). Expected Adaptive Radius: 15.0 km (Standard Mode)\n")
    
    # Send a reset event first
    send_node_payload("ResetNode", 0.0, 0.0, peak_mag=0.1, is_reset=True)
    time.sleep(2.0)

    regional_nodes = [
        {"node_id": "RegNode-1 (Delhi CP)", "lat": 28.6139, "lon": 77.2090},
        {"node_id": "RegNode-2 (Noida Sec 62)", "lat": 28.6280, "lon": 77.3649},
        {"node_id": "RegNode-3 (Ghaziabad)", "lat": 28.6692, "lon": 77.4538},
    ]

    for idx, node in enumerate(regional_nodes):
        print(f"   📡 Transmitting Telemetry for {node['node_id']}...")
        resp = send_node_payload(node["node_id"], node["lat"], node["lon"])
        if resp:
            print(f"      └── Server Response: Alert = {resp.get('alert_level')} | Mw = {resp.get('magnitude_mw', 'N/A')}")
        time.sleep(1.5)

    print("\n   ⏳ Pausing 5 seconds to observe 15.0 km consensus circle on web dashboard...")
    time.sleep(5.0)

    # -------------------------------------------------------------------------
    # PHASE 3: Sparse Suburban / Rural Cluster (Nodes Spaced >18 km Apart)
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("📍 PHASE 3: SPARSE SUBURBAN / RURAL CLUSTER (Outskirts & Regional Borders)")
    print("   Nodes are sparsely distributed (~18.5 km spacing). Expected Adaptive Radius: 25.0 km (Sparse Expanded)\n")

    # Send a reset event first
    send_node_payload("ResetNode", 0.0, 0.0, peak_mag=0.1, is_reset=True)
    time.sleep(2.0)

    sparse_nodes = [
        {"node_id": "SparseNode-1 (Gurugram West)", "lat": 28.4595, "lon": 77.0266},
        {"node_id": "SparseNode-2 (Faridabad South)", "lat": 28.3500, "lon": 77.3100},
        {"node_id": "SparseNode-3 (Greater Noida East)", "lat": 28.4744, "lon": 77.5040},
    ]

    for idx, node in enumerate(sparse_nodes):
        print(f"   📡 Transmitting Telemetry for {node['node_id']}...")
        resp = send_node_payload(node["node_id"], node["lat"], node["lon"])
        if resp:
            print(f"      └── Server Response: Alert = {resp.get('alert_level')} | Mw = {resp.get('magnitude_mw', 'N/A')}")
        time.sleep(1.5)

    print("\n================================================================================")
    print(" 🎉 DEMONSTRATION COMPLETE! Check server terminal logs & web dashboard GIS map.")
    print("================================================================================\n")

if __name__ == "__main__":
    run_adaptive_radius_demo()
