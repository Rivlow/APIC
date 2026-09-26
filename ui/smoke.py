"""Test de fumée sans Qt ni GGUI : python -m ui.smoke (depuis la racine du dépôt)."""
import math
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> int:
    from ui.sim.backend import ensure_taichi
    arch = ensure_taichi()
    from ui.model.project import Project, default_project
    from ui.sim.session import SimSession

    p = default_project()
    s = SimSession()
    s.build(p)
    s.step(p.solver.substeps_per_frame)            # première image : compilation JIT
    s.step(10 * p.solver.substeps_per_frame)
    print(f"[smoke] 10 images de {p.solver.substeps_per_frame} sous-pas : {s.last_step_ms:.0f} ms")
    st = s.stats()
    print(f"[smoke] arch={arch} stats={st}")
    assert st["n_fluid"] > 0 and st["n_solid"] > 0, "particules absentes"
    assert st["t"] > 0 and math.isfinite(st["D_max"]), "état incohérent"

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.json")
        p.save(path)
        assert Project.load(path).to_dict() == p.to_dict(), "aller-retour JSON"

    p.domain.n_grid = 128
    s.build(p)
    s.step(20)
    assert s.stats()["t"] > 0
    s.release()
    assert not s.built and s.x_f is None
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
