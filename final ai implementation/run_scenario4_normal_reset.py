import os
import sys
import subprocess

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_script = os.path.join(script_dir, "simulate_crowd.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print("==================================================")
    print(" 🚀 RUNNING SCENARIO 4: NORMAL AMBIENT NOISE MODE (RESET UI)")
    print("==================================================")
    subprocess.run([sys.executable, "-u", sim_script, "--scenario", "4"], cwd=script_dir, env=env)
