# Solver/Fluid/fluid.py -- Fluide faiblement compressible : p = -E (J - 1), J = det(F) suivi par particule.

import taichi as ti


@ti.data_oriented
class Fluid:

    def __init__(self, n, E, p_mass, p_vol):

        self.x = ti.Vector.field(2, ti.f32, n)
        self.v = ti.Vector.field(2, ti.f32, n)
        self.C = ti.Matrix.field(2, 2, ti.f32, n)       # matrice affine APIC

        self.J = ti.field(ti.f32, n)                    # rapport de volume

        # Constantes de compilation (attributs Python) : figées dans le kernel compilé.
        self.p_mass = p_mass
        self.p_vol = p_vol

        # Paramètre réglable : champ 0-D (un attribut Python serait figé à la compilation).
        self.E = ti.field(ti.f32, shape=())
        self.E[None] = E

    @ti.func
    def stress(self, p):
        # Kirchhoff tau = E (J - 1) I
        return self.E[None] * (self.J[p] - 1.0) * ti.Matrix.identity(ti.f32, 2)

    @ti.func
    def update_deformation(self, p, C_new, dt):
        self.J[p] *= 1.0 + dt * C_new.trace()
