import os
import sys
import subprocess

if __name__ == "__main__":
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final ai implementation")
    pipeline_script = os.path.join(target_dir, "run_pipeline.py")
    print(f"Running automated test pipeline: {pipeline_script}\n")
    res = subprocess.run([sys.executable, pipeline_script] + sys.argv[1:], cwd=target_dir, env=env)
    sys.exit(res.returncode)
