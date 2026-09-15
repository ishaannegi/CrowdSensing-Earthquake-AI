import os
import sys
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_script = os.path.join(script_dir, "simulate_crowd.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print("==================================================")
    print(" 🚀 RUNNING SCENARIO 2: LOCAL CLUSTER CONSENSUS (CONFIRMED ALERT)")
    print("==================================================")
    subprocess.run([sys.executable, "-u", sim_script, "--scenario", "2"], cwd=script_dir, env=env)
