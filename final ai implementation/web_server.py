import http.server
import threading
import json
import queue
import time
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PORT = 5000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# SSE Client Queues
clients = []
clients_lock = threading.Lock()

def broadcast_event(event_data):
    """Pushes a JSON event payload to all connected SSE browser clients."""
    with clients_lock:
        to_remove = []
        for q in clients:
            try:
                q.put_nowait(event_data)
            except Exception:
                to_remove.append(q)
        for q in to_remove:
            if q in clients:
                clients.remove(q)

class SSEHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        # API endpoint for ai_enhanced_detector.py to post live events
        if self.path == "/api/event":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                broadcast_event(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(400)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard" or self.path == "/index.html":
            self.path = "/dashboard.html"
            return super().do_GET()

        if self.path == "/events":
            # Server-Sent Events (SSE) Stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = queue.Queue()
            with clients_lock:
                clients.append(q)

            try:
                # Send initial ping event
                self.wfile.write(b"event: ping\ndata: {}\n\n")
                self.wfile.flush()

                while True:
                    try:
                        data = q.get(timeout=20)
                        json_str = json.dumps(data)
                        msg = f"event: telemetry\ndata: {json_str}\n\n"
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        # Send heartbeat ping every 20s to keep connection alive
                        self.wfile.write(b"event: ping\ndata: {}\n\n")
                        self.wfile.flush()
            except (ConnectionResetError, BrokenPipeError, Exception):
                pass
            finally:
                with clients_lock:
                    if q in clients:
                        clients.remove(q)
            return

        return super().do_GET()

class ThreadedServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        exc_type, _, _ = sys.exc_info()
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return
        super().handle_error(request, client_address)

def start_background_ambient_stream():
    """Generates continuous ambient background telemetry every 1.5s so the dashboard stream is always live."""
    import random
    nodes = [
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
    idx = 0
    while True:
        time.sleep(1.5)
        with clients_lock:
            num_clients = len(clients)
        if num_clients > 0:
            node = nodes[idx % len(nodes)]
            idx += 1
            peak_acc = round(random.uniform(0.08, 0.32), 3)
            sta_lta = round(random.uniform(0.9, 1.4), 2)
            conf = round(random.uniform(0.94, 0.99), 2)
            ambient_pkt = {
                "node_id": node["node_id"],
                "alert_level": "NORMAL",
                "prediction": "ambient_noise",
                "confidence": conf,
                "file": "ambient_telemetry.csv",
                "features": {
                    "peak_mag": peak_acc,
                    "sta_lta_ratio": sta_lta,
                    "duration": 5.0
                },
                "location": {"lat": node["lat"], "lon": node["lon"]}
            }
            broadcast_event(ambient_pkt)

def start_web_server():
    # Start background ambient stream thread
    t = threading.Thread(target=start_background_ambient_stream, daemon=True)
    t.start()

    # Use ThreadingHTTPServer so SSE GET stream does NOT block POST /api/event API calls!
    httpd = ThreadedServer(("", PORT), SSEHandler)
    print(f"==================================================")
    print(f"🖥️  LIVE CROWDSENSING WEB DASHBOARD SERVER")
    print(f"   Open Browser: http://127.0.0.1:{PORT}")
    print(f"==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    start_web_server()

