from taichialgebra import *
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

ti.init(arch=ti.gpu, default_fp=ti.f64)

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


# Offset of the beam from the bottom of the screen
heightoffset = 0.8

#SIMULATION PARAMETERS--------------------------------

n: tint = 80

dt: tfloat = 1e-5

random_particle_position: bool = True

# Number of particles per m^2
particles_per_m2 = 230000

#-----------------------------------------------------

current_time: tfloat = 0.0
frame_dt: tfloat = 1e-3
dx: tfloat = 1.0 / n
inv_dx: tfloat = 1.0 / dx

#BEAM PARAMETERS--------------------------------------

# Beam dimensions (meters)
beam_length: tfloat = 0.4
beam_height: tfloat = 0.05
beam_width: tfloat = dx

# Material density (kg/m^3)
density: tfloat = 10

# Young's modulus (Pa)
E: tfloat = 1e5

# Force on the beam (N/kg)
downforce: tfloat = 9.81 * 2
# -----------------------------------------------------

# Create a filename based on the parameters
prefix = "TEST"

dims = f"[{beam_length}x{beam_height}]"
dens = f"{density}"
# format E as scientific notation
e = '%.0E' % Decimal(f"{E}")
frce = f"{downforce}"
d_t = '%.1E' % Decimal(f"{dt}")
rnd = "T" if random_particle_position else "F"
N = f"{n}"
filename = f"{prefix}{dims}_{dens}_{e}_{frce}_{d_t}_{rnd}_{N}_{particles_per_m2}"

# Check if the file already exists
try:
    with open(f"{filename}.txt", "r") as f:
        print("File already exists, delete if you want to rerun.\nexiting...")
        exit()
except FileNotFoundError:
    ...

vol: tfloat = 1
hardening: tfloat = 100.0
nu: tfloat = 0.37

mu_0: tfloat = E / (2 * (1 + nu))
lambda_0: tfloat = E * nu / ((1 + nu) * (1 - 2 * nu))

total_mass: tfloat = density * beam_length * beam_height * beam_width


particle_count = int(particles_per_m2 * beam_length * beam_height)
gridpoints = ((n + 1)**2) * ((beam_height * beam_length))

cell_volume = dx * dx * dx
particle_mass = total_mass / (gridpoints * cell_volume)

force: tfloat = downforce * total_mass
# Uniform force applied to the beam
w = force / (beam_length)
# Moment of inertia
I = (beam_height ** 3 * beam_width) / 12

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

# Initialize grid
grid = Vec3D.field(shape=((n + 1) * (n + 1),))

# Create a plot and keep the figure and axis objects
fig, ax = plt.subplots()
topavgline, = ax.plot([], [], 'r-', label="Maximum avg deflection at top oscillation")  # Initialize an empty plot
bottomavgline, = ax.plot([], [], 'b-', label="Maximum avg deflection at bottom oscillation")  # Initialize an empty plot
topendline, = ax.plot([], [], 'g-', label="Maximum deflection at free end at top oscillation")  # Initialize an empty plot
bottomendline, = ax.plot([], [], 'y-', label="Maximum deflection at free end at bottom oscillation")  # Initialize an empty plot

# Set the x and y axis labels
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('Average distance from expected deflection (cm)', fontsize=14)
ax.set_title('Distance from expected deflection\n \
over time of top and bottom oscillations', fontsize=14)

@ti.func
def gridIndex(i: tint, j: tint) -> int:
    return i + (n + 1) * j

@ti.kernel
def spawn_rnd_beam():
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

@ti.kernel
def spawn_beam(x: ti.template(), y: ti.template()):
    count = x.shape[0] * y.shape[0]

    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            c = 0x00FF00 if j == 0 else 0xED553B
            c = 0xFF00FF if j == 0 and i == x.shape[0] - 1 else c
            b = True if j == 0 else False
            particles.append(new_Particle(Vec2D(x[i], y[j]), c, b))

    print(f"spawned {count} particles")

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

                if (x < boundary):
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

        oldJ = determinant(F)

        Jp_new = clamp(particles[index].Jp * oldJ / determinant(F), 0.6, 20.0)
        particles[index].Jp = Jp_new
        particles[index].F = F

def update_plot(times, avgpeaks, endpeaks):
    # Create/overwrite a new file and write peaks and peaktimes to it
    with open(f"{filename}.txt", "w") as f:
        for i in range(len(times)):
            f.write(f"{times[i]} {avgpeaks[i]} {endpeaks[i]}\n")

    topavgpeaks = avgpeaks[::2]
    topendpeaks = endpeaks[::2]
    toptimes = times[::2]
    bottomavgpeaks = avgpeaks[1::2]
    bottomendpeaks = endpeaks[1::2]
    bottomtimes = times[1::2]

    topavgline.set_data(toptimes, topavgpeaks)
    bottomavgline.set_data(bottomtimes, bottomavgpeaks)
    topendline.set_data(toptimes, topendpeaks)
    bottomendline.set_data(bottomtimes, bottomendpeaks)

    ax.relim()  # Recompute the data limits
    ax.autoscale_view()  # Rescale the view

    plt.legend()
    plt.draw()  # Update the plot
    plt.pause(0.01)  # Pause briefly to allow the plot to update

if __name__ == '__main__':
    if (random_particle_position):
        spawn_rnd_beam()
    else:
        spacing = np.sqrt((beam_length * beam_height) / particle_count)

        x = np.arange(0.04, beam_length + 0.04, spacing)
        y = np.arange(heightoffset, heightoffset + beam_height, spacing)

        tix = ti.field(dtype=tfloat, shape=(len(x)))
        tiy = ti.field(dtype=tfloat, shape=(len(y)))

        tix.from_numpy(x)
        tiy.from_numpy(y)

        spawn_beam(tix, tiy)

    benchmark_particles = particles.b.to_numpy()
    # get indices of indicies that are True
    benchmark_particle_indicies = [i for i, x in enumerate(benchmark_particles) if x]

    frame: tint = 0
    step: tint = 0

    gui = ti.GUI('Taichi MLS-MPM', res=(window_size, window_size))

    avgpeaks = []
    times = []
    oscillations = []
    endpeaks = []

    while gui.running:
        advance(dt)
        current_time += dt

        if (step % (int(frame_dt / dt)+1)) == 0:
            # Clear screen
            gui.clear(0x112F41)

            # Get particle coordinates and colors
            particle_coords = particles.x.to_numpy()
            pixel_colors = particles.c.to_numpy()

            # remove all pixels and particles that are 0
            pixel_colors = pixel_colors[~(pixel_colors == 0)]
            particle_coords = particle_coords[~(particle_coords == 0).all(1)]

            # Draw grid
            spacing = np.linspace(0, 1, n)
            for space in spacing:
                gui.line([0, space], [1, space], color=0x068587)
                gui.line([space, 0], [space, 1], color=0x068587)

            # Draw support
            gui.line([0.04, 0], [0.04, 1], color=0xFFFFFF)

            # Draw beam
            gui.circles(particle_coords, radius=2, color=pixel_colors)

            # Draw projected deflection
            for x, y, in deflections:
                gui.circle([x, y], radius=2, color=0xFF00FF)

            # Calculate and draw difference from expected deflection
            avg_deflection = 0
            for i in benchmark_particle_indicies:
                [x, y] = particle_coords[i]
                expected_y = heightoffset - yinx(x - 0.04)
                deflection = np.abs(expected_y - y)
                # Avoid visual artifacts
                if (deflection > 0.001):
                    gui.line([x, y], [x, expected_y], color=0xFFFFFF)
                avg_deflection += deflection
            avg_deflection = avg_deflection / len(benchmark_particle_indicies)

            # Get right most particle by getting particle with color 0xFF00FF
            endx = 0
            endy = 0
            for i in range(len(particle_coords)):
                if (pixel_colors[i] == 0xFF00FF):
                    [endx, endy] = particle_coords[i]

            # Draw vertical line at particle
            # gui.line([endx, 0], [endx, 1], color=0xFFFFFF)

            oscillations.append(endy)
            diffs = list(np.diff(oscillations))
            diffs.insert(0, 0)

            if (len(diffs) >= 2):
                if (diffs[-1] * diffs[-2] <= 0):
                    avgpeaks.append(avg_deflection * 100)
                    times.append(current_time)
                    expected_y = heightoffset - yinx(endx - 0.04)
                    enddeflection = np.abs(expected_y - endy)
                    endpeaks.append(enddeflection * 100)
                    gui.show(f"top or bottom{frame}.png")
                    frame+= 1
                    continue

            # Draw the plot
            if (step % ((int(frame_dt / dt)+1) * 100)) == 0:
                update_plot(times, avgpeaks, endpeaks)
                plt.savefig(f"{filename}.png")

            gui.show()
            frame += 1

        step += 1
