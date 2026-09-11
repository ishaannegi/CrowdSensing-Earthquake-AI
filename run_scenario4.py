import os
import sys
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final ai implementation")
    script = os.path.join(target_dir, "simulate_crowd.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print("==================================================")
    print(" 🚀 RUNNING SCENARIO 4: NORMAL AMBIENT NOISE MODE (RESET UI)")
    print("==================================================")
    res = subprocess.run([sys.executable, "-u", script, "--scenario", "4"], cwd=target_dir, env=env)
    sys.exit(res.returncode)
