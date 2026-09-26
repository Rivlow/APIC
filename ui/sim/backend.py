"""Initialisation Taichi, une seule fois par processus, jamais à l'import."""
from __future__ import annotations

_arch: str | None = None


def ensure_taichi(prefer_gpu: bool = True) -> str:
    """Appelle ti.init une seule fois (GPU si possible, sinon CPU) et renvoie le nom de l'arch."""
    global _arch
    if _arch is not None:
        return _arch
    import os

    import taichi as ti

    forced = os.environ.get("APIC_UI_ARCH")          # cuda | vulkan | cpu (diagnostic / choix explicite)
    if forced:
        ti.init(arch=getattr(ti, forced))
    elif prefer_gpu:
        # Vulkan d'abord : c'est la seule arch où GGUI présente une ti.Texture sans passer par le CPU
        # (et la simulation y est aussi rapide que sur CUDA ici). Puis CUDA, puis CPU.
        for arch in (ti.vulkan, ti.cuda, ti.cpu):
            try:
                ti.init(arch=arch)
                break
            except Exception as exc:
                print(f"[ui] arch {arch} indisponible ({exc})")
    else:
        ti.init(arch=ti.cpu)
    _arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    return _arch
