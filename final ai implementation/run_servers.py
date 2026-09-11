import os
import sys
import time
import subprocess
import signal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def get_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("==================================================")
    print(" 🌐 CROWDSENSING LIVE SERVERS (PERSISTENT MODE)   ")
    print("==================================================")

    # 1. Ensure model exists
    model_path = os.path.join(script_dir, "earthquake_ai_model.pkl")
    if not os.path.exists(model_path):
        print("Model not found. Training model on STEAD dataset first...")
        subprocess.run([sys.executable, "ai_trainer.py", "--no-plot"], check=True, env=get_env())

    print("\n[STARTING] Launching Web Dashboard (5000) & TCP Server (9000)...")
    web_process = subprocess.Popen([sys.executable, "web_server.py"], env=get_env())
    server_process = subprocess.Popen([sys.executable, "ai_enhanced_detector.py"], env=get_env())

    time.sleep(2)

    print("\n==================================================")
    print("  ✅ SERVERS ARE LIVE & RUNNING INDEFINITELY!")
    print("==================================================")
    print("  🖥️  Web Dashboard: http://127.0.0.1:5000")
    print("  📡 TCP Server Port: 9000")
    print("\n  👉 Open http://127.0.0.1:5000 in your browser.")
    print("  👉 In a 2nd terminal, run: python simulate_crowd.py")
    print("  👉 Press Ctrl+C in this terminal to stop the servers.")
    print("==================================================\n")

    def shutdown(sig, frame):
        print("\n\n[SHUTDOWN] Stopping Web Dashboard & TCP Server...")
        server_process.terminate()
        web_process.terminate()
        try:
            server_process.wait(timeout=3)
            web_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()
            web_process.kill()
        print("[SUCCESS] All servers stopped cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep running until Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)

if __name__ == "__main__":
    main()
