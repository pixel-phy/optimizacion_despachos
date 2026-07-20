# run_dashboard.py
import subprocess
import sys
import os

if __name__ == "__main__":
    dashboard_dir = os.path.join(os.path.dirname(__file__), 'dashboard')
    os.chdir(dashboard_dir)
    
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd)
