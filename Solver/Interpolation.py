import taichi as ti


@ti.func
def bspline_w(f: ti.f32) -> ti.types.vector(3, ti.f32):  # type: ignore[misc]
    """Quadratic B-spline weights for fractional position f ∈ [0.5, 1.5).
    Stencil: nodes at base, base+1, base+2 where base = floor(x - 0.5).
    D^{-1} = 4/h² exactly (constant, independent of f)."""
    return ti.Vector([
        0.5  * (1.5 - f) ** 2,
        0.75 - (f - 1.0) ** 2,
        0.5  * (f - 0.5) ** 2,
    ])
