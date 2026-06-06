import taichi as ti
import numpy as np

from Boundary_condition.Boundary_condition import *
from Solver.Solver import *
from Solver.incompressibility import *

ti.init(arch=ti.gpu)

#============================#
#     Simulation params      #
#============================#

# Spatial domain
nb_particles = 10_000
nb_grid = 256
dx = 1/nb_grid

# Physical params
p_mass = 1000.0
dt = 1e-3
gravity = ti.Vector([0.0, -9.81])
radius = 0.15  # Radius of the initial particle ball


#=============================#
#     Object definitions      #
#=============================#

pos = ti.Vector.field(2, float, nb_particles)
v = ti.Vector.field(2, float, nb_particles)

C = ti.Matrix.field(2, 2, float, nb_particles) # affine matrix
J = ti.field(float, nb_particles) 

# Grid definition
grid_v = ti.Vector.field(2, float, (nb_grid, nb_grid))
grid_m = ti.field(float, (nb_grid, nb_grid))

# Pressure projection fields
p = ti.field(float, (nb_grid, nb_grid))  # Pressure field
p_new = ti.field(float, (nb_grid, nb_grid))
Ax = ti.field(float, (nb_grid, nb_grid))  # Laplacian(p)
b = ti.field(float, (nb_grid, nb_grid))  # Divergence

# Fluid mask: 1 if cell contains fluid, 0 otherwise
fluid_mask = ti.field(int, (nb_grid, nb_grid))


def substep(step):

    clear_grid(grid_m, grid_v)

    # Step 1: Particle -> grid 
    particle_to_grid(grid_m, grid_v, pos, v, C, dx)

    # Computation on fluid cells 
    apply_fluid_mask(grid_m, fluid_mask)
    normalize_grid(grid_m, grid_v) # m*v -> v (momentum -> velocity)

    # Step 2: apply external forces
    add_gravity(grid_m, grid_v, dt)
    
    # Step 3: ensure incompressibility (pressure projection)
    ensure_incompressibility(grid_m, grid_v, p, p_new, b, dx, dt, fluid_mask)
    
    # Step 4; apply BCs
    apply_boundary(grid_m, grid_v)

    # Step 5: Grid -> particle 
    grid_to_particle(grid_v, pos, v, C, dx)

    # Step 6: Final advection of particles
    advection_particles(pos, v, dt)

if __name__ == "__main__":

    init(pos, radius, v, C)

    gui = ti.GUI("APIC Fluid Simulation", res=512) 

    step = 0
    while gui.running:

        for _ in range(10):
            substep(step)
            step += 1

        gui.circles(pos.to_numpy(), radius=1, color=0x66ccff)
        gui.show()