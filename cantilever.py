from taichialgebra import *
import taichi as ti
import numpy as np

import matplotlib.pyplot as plt

# Create a plot and keep the figure and axis objects
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  # Initialize an empty plot

# Set the x and y axis labels
ax.set_xlabel('Time (s)')
ax.set_ylabel('Average distance from expected deflection (m)')

# #fast_math=False maybe needed for some operations
ti.init(arch=ti.gpu, default_fp=ti.f64)
# ti.init(default_fp=ti.f64)

@ti.dataclass
class Particle:
    x: Vec2D # coord
    v: Vec2D # velocity
    F: Mat
    C: Mat
    Jp: tfloat
    c: tint # color
    b: bool

@ti.func
def new_Particle(x: Vec2D, c: tint, b: bool) -> Particle:
    return Particle(x, Vec2D(0.0, 0.0), Mat(1.0, 0.0, 0.0, 1.0), Mat(0.0, 0.0, 0.0, 0.0), 1.0, c, b)

window_size: tint = 1600
n: tint = 80

dt: tfloat = 1e-5
current_time: tfloat = 0.0
frame_dt: tfloat = 1e-3
dx: tfloat = 1.0 / n
inv_dx: tfloat = 1.0 / dx


particle_mass: tfloat = 585.0
vol: tfloat = 1
hardening: tfloat = 100.0
E: tfloat = 1e6
nu: tfloat = 0.37
plastic: bool = False

mu_0: tfloat = E / (2 * (1 + nu))
lambda_0: tfloat = E * nu / ((1 + nu) * (1 - 2 * nu))

# Beam bending values (meters)
beam_length: tfloat = 0.5
beam_height: tfloat = 0.05
beam_width: tfloat = 1.0/n

heightoffset = 0.8


# Material density (kg/m^3)
density: tfloat = 600.0 / 2
total_mass: tfloat = density * beam_length * beam_height * beam_width

particles_per_cell = 35
gridpoints = ((n + 1)**2) * ((beam_height * beam_length))
particle_count = int(gridpoints * particles_per_cell)

cell_volume = dx * dx * dx
cell_mass = cell_volume * particle_mass
print(gridpoints * cell_mass)
print(total_mass)
good = total_mass / (gridpoints * cell_volume)
print(good)
particle_mass = good
# exit()
# particle_mass = (total_mass * 1000000 / particle_count)
# print(particle_mass, total_mass, particle_mass / total_mass)

downforce: tfloat = 9.81
force: tfloat = downforce * total_mass
# Uniform force applied to the beam
w = force / (beam_length)
# Moment of inertia
I = (beam_height ** 3 * beam_width) / 12

deflection = (w * beam_length ** 4) / (8 * E * I)

print(w, beam_length, E, I, deflection)

def yinx(x):
    return ((w * x ** 2) / (24 * E * I)) * (x ** 2 + 6 * beam_length ** 2 - 4 * beam_length * x)

deflections = []

for x in range(201):
    x = (x / 200) * beam_length
    y = yinx(x)
    deflections.append((x + 0.04, heightoffset - y))

# Initialize list of particles
S = ti.root.dynamic(ti.i, 1024 * 16, chunk_size = 32)
particles = Particle.field()
S.place(particles)

grid = Vec3D.field(shape=((n + 1) * (n + 1),))

@ti.func
def gridIndex(i: tint, j: tint) -> int:
    return i + (n + 1) * j

@ti.kernel
def spawn_single():
    particles.append(new_Particle(Vec2D(0.5, 0.93), 0xED553B))

@ti.kernel
def spawn_beam():
    bottoms = 100

    for _ in range(particle_count - bottoms):
        # x = 0 -> 1
        x = ti.random(dtype=tfloat)
        # scale x from 0.04 to 0.5
        x = x * beam_length + 0.04

        # y = 0 -> 1
        y = ti.random(dtype=tfloat)
        # scale y from 0.5 to 0.55
        y = y * beam_height + heightoffset

        particles.append(new_Particle(Vec2D(x, y), 0xED553B, False))

    for i in range(bottoms):
        x = (i / (bottoms - 1)) * beam_length + 0.04

        y = heightoffset

        c = 0x00FF00 if i != 99 else 0xFF00FF

        particles.append(new_Particle(Vec2D(x, y), c, True))

    print(f"spawned {particle_count} particles")

@ti.func
def add_object(center: Vec2D, c: tint):
    for _ in range(1000):
        # generate random x between 0 and 1
        x = ti.random(dtype=tfloat)
        y = ti.random(dtype=tfloat)
        coord = (Vec2D(x, y) * 2 - Vec2D(1, 1)) * 0.08 + center
        particles.append(new_Particle(coord, c))


@ti.kernel
def advance(dt: tfloat):
    # Reset grid
    grid.fill(Vec3D(0.0, 0.0, 0.0))

    # P2G
    for index in particles:
        p = particles[index]

        base_coord = (p.x * inv_dx - 0.5).cast(tint)
        fx = p.x * inv_dx - base_coord

        w = [
            had2D(Vec2D(0.5, 0.5), (Vec2D(1.5, 1.5) - fx) ** 2),
            Vec2D(0.75, 0.75) - (fx - Vec2D(1.0, 1.0)) ** 2,
            had2D(Vec2D(0.5, 0.5), (fx - Vec2D(0.5, 0.5)) ** 2)
        ]

        e = ti.exp(hardening * (1.0 - p.Jp))
        # e = 1

        mu = mu_0 * e
        lambda_ = lambda_0 * e

        J = determinant(p.F)
        r, s = polar_decomp(p.F)

        k1 = -4 * inv_dx * inv_dx * dt * vol
        k2 = lambda_ * (J - 1) * J

        stress = (mulMat((transposed(p.F) - r),p.F) * 2 * mu + Mat(k2,0,0,k2)) * k1
        affine = stress + p.C * particle_mass

        mv = Vec3D(p.v[0] * particle_mass, p.v[1] * particle_mass, particle_mass)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = Vec2D(float(i) - fx[0], float(j) - fx[1]) * dx
                weight = w[i][0] * w[j][1]
                ii = gridIndex(base_coord[0] + i, base_coord[1] + j)
                mul = mulMatVec(affine, dpos)
                grid[ii] += (mv + Vec3D(mul[0], mul[1], 0.0)) * weight

    # Modify grid velocities to respect boundaries
    boundary = 0.04
    for i in range(n + 1):
        for j in range(n + 1):
            ii = gridIndex(i, j)
            if grid[ii][2] > 0:
                grid[ii] /= grid[ii][2]
                grid[ii] += Vec3D(0, -downforce * dt, 0) # gravity

                x = i / n
                y = j / n

                if (x < boundary or x > 1 - boundary or y > 1 - boundary):
                    grid[ii] = Vec3D(0, 0, 0)

                if (y < boundary):
                    grid[ii][1] = max(0.0, grid[ii][1])

    # G2P
    for index in particles:
        p = particles[index]

        base_coord = (particles[index].x * inv_dx - 0.5).cast(int)
        fx = particles[index].x * inv_dx - base_coord

        w = [
            had2D(Vec2D(0.5, 0.5), (Vec2D(1.5, 1.5) - fx) ** 2),
            Vec2D(0.75, 0.75) - (fx - Vec2D(1.0, 1.0)) ** 2,
            had2D(Vec2D(0.5, 0.5), (fx - Vec2D(0.5, 0.5)) ** 2)
        ]

        particles[index].C = Mat(0.0, 0.0, 0.0, 0.0)
        particles[index].v = Vec2D(0.0, 0.0)

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = Vec2D(i, j) - fx
                weight = w[i][0] * w[j][1]

                ii = gridIndex(base_coord[0] + i, base_coord[1] + j)

                particles[index].v += Vec2D(grid[ii][0], grid[ii][1]) * weight

                particles[index].C += outer_product(Vec2D(grid[ii][0], grid[ii][1]) * weight, dpos) * (4 * inv_dx)

        #advection
        particles[index].x += particles[index].v * dt

        F = mulMat(particles[index].F, (Mat(1.0, 0.0, 0.0, 1.0) + particles[index].C * dt))

        # svd_u, sig, svd_v = svd(F)

        # for i in ti.static(range(2 * int(plastic))):
        #     sig[i + 2*i] = clamp(sig[i+2*i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        oldJ = determinant(F)

        # F = mulMat(mulMat(svd_u, sig), transposed(svd_v))

        Jp_new = clamp(particles[index].Jp * oldJ / determinant(F), 0.6, 20.0)
        particles[index].Jp = Jp_new
        particles[index].F = F

def update_plot(times, distances):
    line.set_data(times, distances)
    ax.relim()  # Recompute the data limits
    ax.autoscale_view()  # Rescale the view
    plt.draw()  # Update the plot
    plt.pause(0.01)  # Pause briefly to allow the plot to update

if __name__ == '__main__':
    spawn_beam()

    # get 100 particles
    benchmark_particles = particles.b.to_numpy()
    # get indices of indicies that are True
    benchmark_particle_indicies = [i for i, x in enumerate(benchmark_particles) if x]

    frame: tint = 0
    last_frame: tint = frame
    step: tint = 0

    gui = ti.GUI('Taichi MLS-MPM', res=(window_size, window_size))

    distances = []
    times = []

    oscillations = []

    while gui.running:
        advance(dt)
        current_time += dt

        if (step % (int(frame_dt / dt)+1)) == 0:
            gui.clear(0x112F41)
            particle_coords = particles.x.to_numpy()
            pixel_colors = particles.c.to_numpy()
            # remove all pixels that are 0
            pixel_colors = pixel_colors[~(pixel_colors == 0)]
            # remove all particles that are [0,0]
            particle_coords = particle_coords[~(particle_coords == 0).all(1)]

            # print(f"Expected deflection: {deflection}m")
            print(f"at {current_time}s")

            spacing = np.linspace(0, 1, n)
            for space in spacing:
                gui.line([0, space], [1, space], color=0x068587)
                gui.line([space, 0], [space, 1], color=0x068587)

            # gui.rect(topleft = [0.04, 0.04], bottomright=[0.96, 0.96], color=0x068587)
            gui.line([0.04, 0], [0.04, 1], color=0xFFFFFF)

            gui.circles(particle_coords, radius=2, color=pixel_colors)

            for x, y, in deflections:
                gui.circle([x, y], radius=2, color=0xFF00FF)

            avg_deflection = 0
            for i in benchmark_particle_indicies:
                [x, y] = particle_coords[i]
                expected_y = heightoffset - yinx(x - 0.04)
                gui.line([x, y], [x, expected_y], color=0xFFFFFF)
                avg_deflection += np.abs(expected_y - y)

            avg_deflection = avg_deflection / len(benchmark_particle_indicies)

            distances.append(avg_deflection)
            times.append(current_time)

            if (step % ((int(frame_dt / dt)+1) * 100)) == 0:
                update_plot(times, distances)

                if (current_time > 1):
                    # save the figure
                    plt.savefig(f"deflections.png")

            # if (frame%5 == 0):
            #     gui.show(f"standard_beam{frame}.png")
            # else:
            gui.show()
            # exit()
            # gui.show(f"crash/{frame}.png")

            frame += 1

        step += 1
