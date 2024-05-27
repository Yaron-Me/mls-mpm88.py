from taichialgebra import *
import taichi as ti
import numpy as np

import matplotlib.pyplot as plt

# Create a plot and keep the figure and axis objects
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  # Initialize an empty plot

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

window_size: tint = 800
n: tint = 80

dt: tfloat = 1e-5
current_time: tfloat = 0.0
frame_dt: tfloat = 1e-3
dx: tfloat = 1.0 / n
inv_dx: tfloat = 1.0 / dx


particle_mass: tfloat = 505.0
vol: tfloat = 1 # scale factor
hardening: tfloat = 100.0
E: tfloat = 3e7
nu: tfloat = 0.37
plastic: bool = False

mu_0: tfloat = E / (2 * (1 + nu))
lambda_0: tfloat = E * nu / ((1 + nu) * (1 - 2 * nu))

# Beam bending values (meters)
beam_length: tfloat = 0.7
beam_height: tfloat = 0.02
beam_width: tfloat = 1.0/n

particle_count = int(beam_length * beam_height * 90000)

# Material density (kg/m^3)
density: tfloat = 600.0
total_mass: tfloat = density * beam_length * beam_height * beam_width

gridpoints = ((n + 1)**2) * ((beam_height * beam_length) / vol)
print(particle_mass, density, gridpoints)
# exit()
# particle_mass = (total_mass * 1000000 / particle_count)
# print(particle_mass, total_mass, particle_mass / total_mass)

force: tfloat = 9.81 * total_mass
# Uniform force applied to the beam
w = force / (beam_length)
# Moment of inertia
I = (beam_height ** 3 * beam_width) / 12

deflection = (w * beam_length ** 4) / (8 * E * I)

print(w, beam_length, E, I, deflection)

deflections = []

for x in range(201):
    x = (x / 200) * beam_length
    y = ((w * x ** 2) / (24 * E * I)) * (x ** 2 + 6 * beam_length ** 2 - 4 * beam_length * x)
    deflections.append((x + 0.04, 0.5 - y))

# Initialize list of particles
S = ti.root.dynamic(ti.i, 1024 * 4, chunk_size = 32)
particles = Particle.field()
S.place(particles)

# Stupid hack to allow for early exit
EXIT = ti.field(ti.i8, shape=())

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
        y = y * beam_height + 0.5

        particles.append(new_Particle(Vec2D(x, y), 0xED553B, False))

    for i in range(bottoms):
        x = (i / (bottoms - 1)) * beam_length + 0.04

        y = 0.5

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
        # cauchy = mu * (mulMat(p.F, transposed(p.F))) + Mat(1, 0, 0, 1) * (lambda_ * ti.log(J) - mu)
        # stress = -(dt * vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p.C * particle_mass

        mv = Vec3D(p.v[0] * particle_mass, p.v[1] * particle_mass, particle_mass)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = Vec2D(float(i) - fx[0], float(j) - fx[1]) * dx
                weight = w[i][0] * w[j][1]
                ii = gridIndex(base_coord[0] + i, base_coord[1] + j)
                mul = mulMatVec(affine, dpos)
                # ti.atomic_add(grid[ii], (mv + Vec3D(mul[0], mul[1], 0.0)) * weight)
                grid[ii] += (mv + Vec3D(mul[0], mul[1], 0.0)) * weight

    # Modify grid velocities to respect boundaries
    boundary = 0.04
    for i in range(n + 1):
        for j in range(n + 1):
            ii = gridIndex(i, j)
            if grid[ii][2] > 0:
                grid[ii] /= grid[ii][2]
                grid[ii] += Vec3D(0, -9.81 * dt / vol, 0) # gravity

                x = i / n
                y = j / n

                if (x < boundary or x > 1 - boundary or y > 1 - boundary):
                    grid[ii] = Vec3D(0, 0, 0)

                if (y < boundary):
                    EXIT[None] = ti.cast(1, ti.i8)
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

@ti.kernel
def initEXIT():
    EXIT[None] = ti.cast(0, ti.i8)

@ti.kernel
def smallestDistances(points1: ti.template(), points2: ti.template()) -> tfloat:
    n = points1.shape[0]
    m = points2.shape[0]
    totalDistances = 0.0
    for i in range(n):
        smallestDistance = 100.0
        for j in range(m):
            distance = (Vec2D(points1[i, 0],points1[i,1] - Vec2D(points1[j, 0],points1[j,1]))).norm()
            if distance < smallestDistance:
                smallestDistance = distance
        totalDistances += smallestDistance
    return totalDistances/n


def find_closest_points(array1, array2):
    # array1 is of shape (m, 2) and array2 is of shape (n, 2)
    # We will broadcast the subtraction and compute distances
    diff = array1[:, np.newaxis, :] - array2[np.newaxis, :, :]  # shape (m, n, 2)
    distances = np.sum(diff**2, axis=2)  # shape (m, n), squared distances
    closest_indices = np.argmin(distances, axis=1)  # shape (m,), index of the closest point in array2 for each point in array1
    closest_points = array2[closest_indices]  # shape (m, 2), the closest points
    closest_distances = np.sqrt(distances[np.arange(distances.shape[0]), closest_indices])  # shape (m,), the distances to the closest points

    return closest_points, closest_indices, closest_distances

def update_plot(times, distances):
    line.set_data(times, distances)
    ax.relim()  # Recompute the data limits
    ax.autoscale_view()  # Rescale the view
    plt.draw()  # Update the plot
    plt.pause(0.01)  # Pause briefly to allow the plot to update

if __name__ == '__main__':
    initEXIT()
    print(EXIT[None])
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
        if EXIT[None] == 1:
            print(f"reached bottom at {current_time}")
            exit()
        current_time += dt

        if (step % int(frame_dt / dt)) == 0:
            gui.clear(0x112F41)
            particle_coords = particles.x.to_numpy()
            pixel_colors = particles.c.to_numpy()
            # remove all pixels that are 0
            pixel_colors = pixel_colors[~(pixel_colors == 0)]
            # remove all particles that are [0,0]
            particle_coords = particle_coords[~(particle_coords == 0).all(1)]

            # print(f"Expected deflection: {deflection}m")
            print(f"at {current_time}s")
            gui.circles(particle_coords, radius=2, color=pixel_colors)

            gui.rect(topleft = [0.04, 0.04], bottomright=[0.96, 0.96], color=0x068587)
            # draw lines every 10cm
            for y in range(10):
                y = y / 10
                # draw line at y
                gui.line([0.04, y], [0.96, y], color=0xFFFFFF)

            for x, y, in deflections: # [0:np.ceil(100*(far_x  - 0.04)/beam_length)]
                gui.circle([x, y], radius=2, color=0x00FF00)

            # draw horizontal line at min_y
            gui.line([0.04, 0.5 - deflection], [0.96, 0.5 - deflection], color=0x00FF00)

            benchmark_coords = []
            for i in benchmark_particle_indicies:
                [x, y] = particle_coords[i]
                benchmark_coords.append(Vec2D(x, y))

            benchmark_coords = np.array(benchmark_coords)
            benchmark_coords2 = np.array(deflections)

            (closest_points, closest_indices, closest_distances) = find_closest_points(benchmark_coords, benchmark_coords2)

            avg_deflection = 0
            for p1, p2 in zip(benchmark_coords, closest_points):
                gui.line(p1, p2, color=0xFFFFFF)
                avg_deflection += (p1[1] - p2[1])
            avg_deflection = avg_deflection / len(benchmark_coords)

            distances.append(avg_deflection)
            times.append(current_time)

            if (step % (int(frame_dt / dt) * 100)) == 0:
                update_plot(times, distances)

            gui.show()

            frame += 1

        step += 1
