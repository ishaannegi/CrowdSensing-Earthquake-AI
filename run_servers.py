import os
import sys
import subprocess

if __name__ == "__main__":
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final ai implementation")
    script = os.path.join(target_dir, "run_servers.py")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"Starting persistent crowdsensing servers: {script}\n")
    res = subprocess.run([sys.executable, script] + sys.argv[1:], cwd=target_dir, env=env)
    sys.exit(res.returncode)
