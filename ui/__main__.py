"""Point d'entrée : `python -m ui [simulation.json]`."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)                     # Code_tuto/ est un package « namespace »

from ui.runner import SimulationRunner  # noqa: E402

path = next((a for a in sys.argv[1:] if a.lower().endswith(".json")), None)
runner = SimulationRunner.load(path) if path else SimulationRunner.demo()
raise SystemExit(runner.show(path))
