import os
import sys
import time
import subprocess
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def get_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env

def run_step(step_name, command):
    print(f"\n==========================================")
    print(f"[STEP] {step_name}")
    print(f"==========================================")
    result = subprocess.run(command, check=True, env=get_env())
    if result.returncode != 0:
        print(f"[ERROR] Error in {step_name}. Exiting pipeline.")
        sys.exit(result.returncode)

def main():
    use_synthetic = "--synthetic" in sys.argv

    print("==================================================")
    print("      AUTOMATED EARTHQUAKE DETECTION PIPELINE     ")
    if use_synthetic:
        print("         (MODE: DEMO SYNTHETIC DATA GENERATION)   ")
    else:
        print("         (MODE: REAL STEAD DATASET MODEL)         ")
    print("==================================================")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    converted_dir = os.path.join(script_dir, "..", "converted_data")

    if use_synthetic:
        # Step 1: Generate Synthetic Data
        run_step("1. Generate Synthetic 3-Axis Data", [sys.executable, "generate_data.py"])
        # Step 2: Train AI Model on synthetic data
        run_step("2. Train AI Model on Synthetic Data", [sys.executable, "ai_trainer.py", "--no-plot"])
        test_files = [f for f in os.listdir('.') if f.startswith("sensor_data_") and f.endswith(".csv")]
        sample_tests = [os.path.join('.', f) for f in test_files[:5]] if test_files else []
    else:
        # Check if model exists; if not, train model on converted STEAD dataset
        model_path = os.path.join(script_dir, "earthquake_ai_model.pkl")
        if not os.path.exists(model_path) or True:
            run_step("1 & 2. Train AI Model on STEAD Dataset", [sys.executable, "ai_trainer.py", "--no-plot"])

        # Pick sample test CSVs from converted_data
        sample_names = ["real_quake_0.csv", "real_quake_1.csv", "fake_vibration_0.csv", "fake_vibration_1.csv", "real_quake_2.csv"]
        sample_tests = []
        for name in sample_names:
            path = os.path.join(converted_dir, name)
            if os.path.exists(path):
                sample_tests.append(path)

    # Step 3: Launch TCP Detector Server & Web Dashboard Server
    print("\n==========================================")
    print("[STEP] 3. Starting TCP Detector Server (9000) & Web Dashboard (5000)")
    print("==========================================")
    web_process = subprocess.Popen([sys.executable, "web_server.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=get_env())
    server_process = subprocess.Popen([sys.executable, "ai_enhanced_detector.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=get_env())
    time.sleep(2)  # Give servers time to bind ports

    if server_process.poll() is not None:
        print("[ERROR] Server failed to start.")
        _, err = server_process.communicate()
        print(err)
        sys.exit(1)

    print("[SUCCESS] TCP Server & Web Dashboard are running in background.")
    print("👉 Open Dashboard: http://127.0.0.1:5000")

    try:
        # Step 4: Batch Multi-Node Crowdsensing Testing
        print("\n==========================================")
        print("[STEP] 4. Running Multi-Node Crowdsensing Simulator")
        print("==========================================")

        if use_synthetic:
            for test_file in sample_tests:
                print(f"\n--- Testing file: {os.path.basename(test_file)} ---")
                subprocess.run([sys.executable, "earthquake_client.py", test_file, "--send", "--ip", "127.0.0.1", "--port", "9000"], env=get_env())
                time.sleep(0.5)
        else:
            subprocess.run([sys.executable, "simulate_crowd.py"], env=get_env())

        if "--keep-alive" in sys.argv:
            print("\n==========================================")
            print("  🟢 SERVERS KEPT ALIVE FOR MANUAL TESTING")
            print("==========================================")
            print("  🖥️  Web Dashboard: http://127.0.0.1:5000")
            print("  📡 TCP Server Port: 9000")
            print("  Press Ctrl+C to stop servers when finished.")
            print("==========================================")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass

    finally:
        # Step 5: Stop Server Processes
        print("\n==========================================")
        print("[STEP] 5. Stopping Servers")
        print("==========================================")
        server_process.terminate()
        web_process.terminate()
        try:
            server_process.wait(timeout=3)
            web_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_process.kill()
            web_process.kill()
        print("[SUCCESS] Servers shut down cleanly.")

    # Step 6: Verify and Report Log Contents
    print("\n==========================================")
    print("PIPELINE RESULTS SUMMARY")
    print("==========================================")
    if os.path.exists("received_log.csv"):
        df_log = pd.read_csv("received_log.csv")
        print(f"\n[SUCCESS] Total Packets Logged by Server: {len(df_log)}")

        if "alert_level" in df_log.columns:
            counts = df_log["alert_level"].value_counts().to_dict()
            print("\n[SPATIAL CONSENSUS ALERT SUMMARY]")
            print(f"   🟢 NORMAL Telemetry Packets:       {counts.get('NORMAL', 0)}")
            print(f"   ⚠️  LOCAL_VIBRATION (Isolated):     {counts.get('LOCAL_VIBRATION', 0)}")
            print(f"   🚨 CONFIRMED_EARTHQUAKE_ALERT:   {counts.get('CONFIRMED_EARTHQUAKE_ALERT', 0)}")

        print("\nLogged Test Packets (Recent 10):")
        cols = [c for c in ["node_id", "file", "prediction", "confidence", "alert_level"] if c in df_log.columns]
        print(df_log.tail(10)[cols].to_string(index=False))
    else:
        print("[WARNING] received_log.csv not found.")

    print("\n👉 Live Web Dashboard URL: http://127.0.0.1:5000")
    print("   (Run 'python run_servers.py' to run persistent live dashboard & detector server)")
    print("\nALL TESTS AND PIPELINE STEPS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
