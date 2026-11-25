"""Make ``src/`` importable when scripts are run from the repository root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
