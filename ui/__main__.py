"""Point d'entrée : `python -m ui` depuis n'importe quel cwd."""
import os
import sys

# Les bibliothèques de kernels (APIC/, Code_tuto/) sont des packages « namespace » : il faut la racine
# du dépôt (parent de ui/) dans sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ui.app import main  # noqa: E402

raise SystemExit(main())
