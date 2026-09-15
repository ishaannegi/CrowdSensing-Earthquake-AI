import socket
import json
import os
import sys
import math
import logging
import threading
import joblib
import pandas as pd
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SeismicServer")

def load_active_model():
    manifest_paths = [
        "model_manifest.json",
        "final ai implementation/model_manifest.json",
        os.path.join(os.path.dirname(__file__), "model_manifest.json"),
        os.path.join(os.path.dirname(__file__), "..", "model_manifest.json")
    ]
    for m_path in manifest_paths:
        if os.path.exists(m_path):
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                    versioned = manifest.get("versioned_filename", "earthquake_ai_model.pkl")
                    m_dir = os.path.dirname(m_path)
                    candidate = os.path.join(m_dir, versioned) if m_dir else versioned
                    if os.path.exists(candidate):
                        logger.info(f"🔑 [MANIFEST RESOLUTION] Server Active Model Version '{manifest.get('active_version', 'v1')}' -> {candidate}")
                        return joblib.load(candidate)
            except Exception as e:
                logger.warning(f"Error reading model_manifest.json: {e}")

    fallback = os.path.join(os.path.dirname(__file__), "earthquake_ai_model.pkl")
    if not os.path.exists(fallback):
        fallback = "earthquake_ai_model.pkl"
    logger.info(f"🔑 [MANIFEST FALLBACK] Loading model file -> {fallback}")
    try:
        return joblib.load(fallback)
    except Exception as e:
        logger.error(f"Failed to load fallback model {fallback}: {e}")
        return None

ACTIVE_MODEL = None


HOST = "0.0.0.0"
PORT = 9000
LOGFILE = "received_log.csv"
CONSENSUS_WINDOW_SEC = 10.0  # 10s rolling consensus window for multi-node arrival
RADIUS_KM = 15.0  # 15km spatial consensus clustering radius

V_P = 6.0  # Primary (compressional) wave velocity in km/s
V_S = 3.5  # Secondary (shear) wave velocity in km/s

event_buffer = []
buffer_lock = threading.Lock()
log_lock = threading.Lock()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def calculate_lead_time(distance_km):
    if distance_km <= 0:
        return 0.0
    return round(distance_km * ((1.0 / V_S) - (1.0 / V_P)), 1)

def retroactive_update_log(updated_node_ids, epicenter_lat=0.0, epicenter_lon=0.0):
    sc_files = [f"received_log_scenario{i}.csv" for i in range(1, 5)]
    target_files = [LOGFILE] + sc_files
    for target in target_files:
        if not os.path.exists(target):
            continue
        try:
            with log_lock:
                df = pd.read_csv(target)
                if "node_id" in df.columns and "alert_level" in df.columns and "prediction" in df.columns:
                    mask = (df["node_id"].isin(updated_node_ids)) & (df["prediction"] == "real_quake") & (df["alert_level"] == "LOCAL_VIBRATION")
                    if mask.any():
                        df.loc[mask, "alert_level"] = "CONFIRMED_EARTHQUAKE_ALERT"
                        if "lead_time_sec" in df.columns and "lat" in df.columns and "lon" in df.columns:
                            for idx in df[mask].index:
                                n_lat, n_lon = df.loc[idx, "lat"], df.loc[idx, "lon"]
                                n_dist = haversine(n_lat, n_lon, epicenter_lat, epicenter_lon)
                                df.loc[idx, "lead_time_sec"] = calculate_lead_time(n_dist)
                        df.to_csv(target, index=False)
                        logger.info(f"🔄 [RETROACTIVE LOG UPDATE] Updated {target} for nodes: {list(updated_node_ids)}")
        except Exception as e:
            logger.error(f"Retroactive log update error for {target}: {e}")

ALL_NODES_REGISTRY = [
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

def estimate_magnitude_mw(cluster_records, epicenter_lat, epicenter_lon):
    """
    Peak Displacement & Acceleration Magnitude Scaling Engine (Wu & Kanamori 2005):
    Mw = 0.78 * log10(Pd_cm) + 1.15 * log10(R) + 3.8
    where Pd_cm is peak ground acceleration in cm/s^2, and R is epicentral distance in km.
    """
    p_d_vals = [ev.get("peak_mag", 2.5) for ev in cluster_records if ev.get("peak_mag", 0.0) > 0]
    avg_pd = float(sum(p_d_vals) / len(p_d_vals)) if p_d_vals else 2.5
    
    distances = [haversine(ev["lat"], ev["lon"], epicenter_lat, epicenter_lon) for ev in cluster_records]
    avg_r = max(float(sum(distances) / len(distances)), 1.0)
    
    pd_cm = max(avg_pd * 10.0, 0.1) # Convert m/s^2 to cm/s^2 scale
    raw_mw = 0.78 * math.log10(pd_cm) + 1.15 * math.log10(avg_r) + 3.8
    mw = round(max(min(raw_mw, 8.0), 3.5), 1)

    logger.info("🧮 [MATH ENGINE] Real-Time Moment Magnitude (Mw) Calculation Breakdown:")
    logger.info(f"   ├── Cluster Peak Accelerations (Pd): {[round(v, 2) for v in p_d_vals]} m/s² -> Avg Pd: {avg_pd:.2f} m/s² ({pd_cm:.2f} cm/s²)")
    logger.info(f"   ├── Cluster Epicentral Distances (R): {[round(d, 2) for d in distances]} km -> Avg R: {avg_r:.2f} km")
    logger.info(f"   ├── Empirical Equation: Mw = 0.78 * log10({pd_cm:.2f}) + 1.15 * log10({avg_r:.2f}) + 3.80")
    logger.info(f"   └── Calculated Result: Mw = {mw} (Moment Magnitude)")

    return mw

def compute_adaptive_radius(curr_lat, curr_lon, event_buffer):
    """
    Dynamic Density-Adaptive Spatial Radius Scaling Engine (DBSCAN / Spatial Nearest-Neighbor Scaling):
    Evaluates local spatial node density around (curr_lat, curr_lon).
    - Dense Urban Node Density (mean k-NN dist <= 3.5 km): Tightens consensus radius to 5.0 km for high spatial precision.
    - Standard Node Density (3.5 km < mean k-NN dist <= 10.0 km): Standard 15.0 km consensus radius.
    - Sparse Suburban Node Density (mean k-NN dist > 10.0 km): Expands consensus radius to 25.0 km to cluster sparse arrays.
    """
    active_coords = [(ev["lat"], ev["lon"]) for ev in event_buffer if ev.get("prediction") == "real_quake"]
    active_coords.append((curr_lat, curr_lon))
    
    if len(active_coords) <= 1:
        return 15.0, "Standard Regional (15km)"

    dists = [haversine(curr_lat, curr_lon, lat, lon) for lat, lon in active_coords if (lat, lon) != (curr_lat, curr_lon)]
    if not dists:
        return 15.0, "Standard Regional (15km)"
    
    mean_knn_dist = float(sum(dists) / len(dists))

    if mean_knn_dist <= 3.5:
        radius_km = 5.0
        mode = "Urban High-Density (5km)"
    elif mean_knn_dist <= 10.0:
        radius_km = 15.0
        mode = "Standard Regional (15km)"
    else:
        radius_km = 25.0
        mode = "Sparse Suburban (25km Expanded)"

    logger.info(f"🌐 [ADAPTIVE SPATIAL RADIUS ENGINE] Local Node Density: Mean k-NN Spacing = {mean_knn_dist:.2f} km -> Scaling Consensus Radius to {radius_km} km ({mode})")
    return radius_km, mode

def compute_consensus(curr_node, curr_lat, curr_lon, curr_pred, now_ts, peak_mag=2.5):
    global event_buffer

    with buffer_lock:
        # Clean up buffer: remove events older than CONSENSUS_WINDOW_SEC (10.0s)
        event_buffer = [ev for ev in event_buffer if (now_ts - ev["timestamp"]) <= CONSENSUS_WINDOW_SEC]

        adaptive_radius_km, density_mode = compute_adaptive_radius(curr_lat, curr_lon, event_buffer)

        if curr_pred != "real_quake":
            event_buffer.append({
                "timestamp": now_ts,
                "node_id": curr_node,
                "lat": curr_lat,
                "lon": curr_lon,
                "prediction": curr_pred,
                "peak_mag": peak_mag,
                "alert_level": "NORMAL",
                "lead_time_sec": 0.0
            })
            return "NORMAL", 0.0, None

        # Collect all nearby real_quake node records within adaptive_radius_km
        cluster_records = []
        for ev in event_buffer:
            if ev["prediction"] == "real_quake":
                dist = haversine(curr_lat, curr_lon, ev["lat"], ev["lon"])
                if dist <= adaptive_radius_km:
                    cluster_records.append(ev)

        cluster_records.append({
            "timestamp": now_ts,
            "node_id": curr_node,
            "lat": curr_lat,
            "lon": curr_lon,
            "peak_mag": peak_mag,
            "prediction": curr_pred
        })

        unique_nodes = set(ev["node_id"] for ev in cluster_records)
        count = len(unique_nodes)
        lead_time_sec = 0.0
        epicenter_info = None

        if count >= 3:
            alert_level = "CONFIRMED_EARTHQUAKE_ALERT"
            epicenter_lat = sum(ev["lat"] for ev in cluster_records) / len(cluster_records)
            epicenter_lon = sum(ev["lon"] for ev in cluster_records) / len(cluster_records)

            mw_val = estimate_magnitude_mw(cluster_records, epicenter_lat, epicenter_lon)

            cluster_nodes = [{"id": ev["node_id"], "lat": ev["lat"], "lon": ev["lon"]} for ev in cluster_records]
            epicenter_info = {
                "epicenter": {"lat": round(epicenter_lat, 4), "lon": round(epicenter_lon, 4)},
                "cluster_nodes": cluster_nodes,
                "magnitude_mw": mw_val,
                "adaptive_radius_km": adaptive_radius_km,
                "density_mode": density_mode
            }

            curr_dist = haversine(curr_lat, curr_lon, epicenter_lat, epicenter_lon)
            lead_time_sec = calculate_lead_time(curr_dist)

            updated_nodes = []
            for ev in event_buffer:
                if ev["node_id"] in unique_nodes and ev["prediction"] == "real_quake":
                    if ev.get("alert_level") != "CONFIRMED_EARTHQUAKE_ALERT":
                        ev["alert_level"] = "CONFIRMED_EARTHQUAKE_ALERT"
                        node_dist = haversine(ev["lat"], ev["lon"], epicenter_lat, epicenter_lon)
                        ev["lead_time_sec"] = calculate_lead_time(node_dist)
                        updated_nodes.append(ev["node_id"])

            if updated_nodes:
                retroactive_update_log(updated_nodes, epicenter_lat, epicenter_lon)
                for ev in event_buffer:
                    if ev["node_id"] in updated_nodes:
                        retro_payload = {
                            "node_id": ev["node_id"],
                            "location": {"lat": ev["lat"], "lon": ev["lon"]},
                            "prediction": ev["prediction"],
                            "confidence": 0.95,
                            "lead_time_sec": ev.get("lead_time_sec", 0.0),
                            "features": {"peak_mag": ev.get("peak_mag", 2.5), "sta_lta_ratio": 4.0},
                            "file": "retroactive_consensus_update",
                            "epicenter": epicenter_info["epicenter"],
                            "cluster_nodes": epicenter_info["cluster_nodes"],
                            "magnitude_mw": mw_val
                        }
                        push_to_web_dashboard(retro_payload, "CONFIRMED_EARTHQUAKE_ALERT", ev.get("lead_time_sec", 0.0))

            predictive_warnings = []
            for reg_node in ALL_NODES_REGISTRY:
                is_in_cluster = any(ev["node_id"] == reg_node["node_id"] or reg_node["node_id"] in ev["node_id"] for ev in cluster_records)
                if not is_in_cluster:
                    dist_km = haversine(reg_node["lat"], reg_node["lon"], epicenter_lat, epicenter_lon)
                    pred_lead_time = calculate_lead_time(dist_km)
                    predictive_warnings.append({
                        "node_id": reg_node["node_id"],
                        "lat": reg_node["lat"],
                        "lon": reg_node["lon"],
                        "distance_km": round(dist_km, 1),
                        "lead_time_sec": pred_lead_time,
                        "magnitude_mw": mw_val,
                        "status_msg": f"⚠️ {reg_node['node_id'].split(' ')[0]}: M{mw_val} shaking in {pred_lead_time:.1f}s (not yet detected locally)"
                    })

            if predictive_warnings:
                warning_payload = {
                    "event_type": "incoming_warning",
                    "node_id": curr_node,
                    "epicenter": {"lat": round(epicenter_lat, 4), "lon": round(epicenter_lon, 4)},
                    "cluster_nodes": cluster_nodes,
                    "magnitude_mw": mw_val,
                    "warnings": predictive_warnings
                }
                push_to_web_dashboard(warning_payload, "CONFIRMED_EARTHQUAKE_ALERT", lead_time_sec)
        else:
            alert_level = "LOCAL_VIBRATION"

        event_buffer.append({
            "timestamp": now_ts,
            "node_id": curr_node,
            "lat": curr_lat,
            "lon": curr_lon,
            "prediction": curr_pred,
            "peak_mag": peak_mag,
            "alert_level": alert_level,
            "lead_time_sec": lead_time_sec
        })

        return alert_level, lead_time_sec, epicenter_info

def log_to_csv(payload, alert_level, lead_time_sec=0.0):
    scenario_num = payload.get("scenario")
    targets = [LOGFILE]
    if scenario_num:
        targets.append(f"received_log_scenario{scenario_num}.csv")

    location = payload.get("location", {})
    lat = location.get("lat", 0.0)
    lon = location.get("lon", 0.0)
    node_id = payload.get("node_id", "Unknown-Node")

    line = "{timestamp},{node_id},{lat},{lon},{file},{prediction},{confidence},{peak_mag},{sta_lta_ratio},{lead_time_sec},{alert_level}\n".format(
        timestamp=payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
        node_id=node_id,
        lat=lat,
        lon=lon,
        file=os.path.basename(str(payload.get("file", ""))),
        prediction=payload.get("prediction", ""),
        confidence=float(payload.get("confidence", 0.0)),
        peak_mag=float(payload.get("features", {}).get("peak_mag", 0.0)),
        sta_lta_ratio=float(payload.get("features", {}).get("sta_lta_ratio", 0.0)),
        lead_time_sec=lead_time_sec if alert_level == "CONFIRMED_EARTHQUAKE_ALERT" else 0.0,
        alert_level=alert_level
    )

    with log_lock:
        for target in targets:
            file_exists = os.path.exists(target)
            try:
                with open(target, "a", encoding="utf-8") as f:
                    if not file_exists:
                        f.write("timestamp,node_id,lat,lon,file,prediction,confidence,peak_mag,sta_lta_ratio,lead_time_sec,alert_level\n")
                    f.write(line)
            except Exception as e:
                logger.error(f"Error writing log {target}: {e}")

import urllib.request

def push_to_web_dashboard(payload, alert_level, lead_time_sec=0.0):
    try:
        data = dict(payload)
        data["alert_level"] = alert_level
        data["lead_time_sec"] = lead_time_sec
        req = urllib.request.Request(
            "http://127.0.0.1:5000/api/event",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            pass
    except Exception as e:
        logger.warning(f"Dashboard push warning (ensure web_server.py on port 5000 is running): {e}")

def handle_client(conn, addr):
    conn.settimeout(5.0)
    try:
        data = conn.recv(16384)
        if not data:
            return

        payload = json.loads(data.decode("utf-8"))
        now_ts = datetime.now(timezone.utc).timestamp()

        # Server-side model inference using active model loaded from manifest
        if ACTIVE_MODEL is not None and "features" in payload:
            feats = payload["features"]
            if isinstance(feats, dict):
                feat_vector = [
                    float(feats.get("duration", 0.0)),
                    float(feats.get("peak_mag", 0.0)),
                    float(feats.get("rms_mag", 0.0)),
                    float(feats.get("sta_lta_ratio", 0.0)),
                    float(feats.get("dom_freq", 0.0)),
                    float(feats.get("peak_ax", 0.0)),
                    float(feats.get("peak_ay", 0.0)),
                    float(feats.get("peak_az", 0.0)),
                    float(feats.get("rms_ax", 0.0)),
                    float(feats.get("rms_ay", 0.0)),
                    float(feats.get("rms_az", 0.0))
                ]
                try:
                    pred_res = str(ACTIVE_MODEL.predict([feat_vector])[0])
                    try:
                        proba = ACTIVE_MODEL.predict_proba([feat_vector])[0]
                        conf_res = round(float(proba[list(ACTIVE_MODEL.classes_).index(pred_res)]), 4)
                    except Exception:
                        conf_res = 0.0
                    payload["prediction"] = pred_res
                    payload["confidence"] = conf_res
                except Exception as e:
                    logger.warning(f"Server AI inference error: {e}")

        node_id = payload.get("node_id", f"Client-{addr[0]}")
        loc = payload.get("location", {})
        lat = float(loc.get("lat", 0.0))
        lon = float(loc.get("lon", 0.0))
        pred = payload.get("prediction", "fake_vibration")
        conf = float(payload.get("confidence", 0.0))
        filename = os.path.basename(str(payload.get("file", "")))
        peak_mag = float(payload.get("features", {}).get("peak_mag", 2.5))

        alert_level, lead_time_sec, epicenter_info = compute_consensus(node_id, lat, lon, pred, now_ts, peak_mag=peak_mag)
        if epicenter_info:
            payload.update(epicenter_info)

        log_to_csv(payload, alert_level, lead_time_sec)
        push_to_web_dashboard(payload, alert_level, lead_time_sec)

        mw_str = f" | Mw: {epicenter_info.get('magnitude_mw')}" if epicenter_info and "magnitude_mw" in epicenter_info else ""
        logger.info(f"📩 [TELEMETRY] Node: {node_id} ({lat:.4f}, {lon:.4f}) | Sample: {filename} | AI: {pred} ({conf:.2%}) | Alert: {alert_level}{mw_str}")

        response_dict = {
            "status": "OK",
            "alert_level": alert_level,
            "lead_time_sec": lead_time_sec,
            "node_id": node_id
        }
        if epicenter_info and "magnitude_mw" in epicenter_info:
            response_dict["magnitude_mw"] = epicenter_info["magnitude_mw"]

        response = json.dumps(response_dict).encode("utf-8")
        conn.sendall(response)

    except (socket.timeout, socket.error, ConnectionResetError, BrokenPipeError) as e:
        logger.warning(f"Socket error handling client {addr}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON payload from {addr}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error handling client {addr}: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def start_server():
    global ACTIVE_MODEL
    if ACTIVE_MODEL is None:
        ACTIVE_MODEL = load_active_model()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(20)
        logger.info(f"==================================================")
        logger.info(f"🌐 MULTI-THREADED CROWDSENSING TCP SERVER STARTED")
        logger.info(f"   Listening on {HOST}:{PORT}")
        logger.info(f"   Spatial Radius: {RADIUS_KM} km | Time Window: {CONSENSUS_WINDOW_SEC}s")
        logger.info(f"==================================================")

        while True:
            try:
                conn, addr = s.accept()
                threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
            except KeyboardInterrupt:
                logger.info("Server shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Accept error: {e}")

if __name__ == "__main__":
    start_server()
