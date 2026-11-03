import numpy as np
import subprocess
import sys
import time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
Path(__file__).with_name("child_pid.txt").write_text(str(child.pid), encoding="utf-8")
time.sleep(300)
def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    return np.ones_like(distance_matrix)
