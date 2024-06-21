from taichialgebra import *
import taichi as ti

#fast_math=False maybe needed for some operations
ti.init(arch=ti.gpu, default_fp=ti.f64)
# ti.init(default_fp=ti.f64)

@ti.dataclass
class Particle:
    x: Vec2D
    v: Vec2D
    F: Mat
    C: Mat
    Jp: tfloat
    c: tint

@ti.func
def new_Particle(x: Vec2D, c: tint) -> Particle:
    return Particle(x, Vec2D(0.0, 0.0), Mat(1.0, 0.0, 0.0, 1.0), Mat(0.0, 0.0, 0.0, 0.0), 1.0, c)

window_size: tint = 800
n: tint = 80

dt: tfloat = 1e-4
frame_dt: tfloat = 1e-3
dx: tfloat = 1.0 / n
inv_dx: tfloat = 1.0 / dx

particle_mass: tfloat = 1.0
vol: tfloat = 1.0
hardening: tfloat = 10.0
E: tfloat = 1e4
nu: tfloat = 0.2
plastic: bool = True

mu_0: tfloat = E / (2 * (1 + nu))
lambda_0: tfloat = E * nu / ((1 + nu) * (1 - 2 * nu))

# Initialize list of particles
S = ti.root.dynamic(ti.i, 1024 * 16, chunk_size = 32)
particles = Particle.field()
S.place(particles)

grid = Vec3D.field(shape=((n + 1) * (n + 1),))

@ti.func
def gridIndex(i: tint, j: tint) -> int:
    return i + (n + 1) * j

@ti.kernel
def add_random_particles():
    color = int(ti.random() * 0xFFFFFF)
    add_object(Vec2D(0.5,0.8), color)

@ti.kernel
def init_particles():
    add_object(Vec2D(0.55, 0.45), 0xED553B)
    add_object(Vec2D(0.45,0.65), 0xF2B134)
    add_object(Vec2D(0.55,0.85), 0x068587)

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
    boundary = 0.05
    for i in range(n + 1):
        for j in range(n + 1):
            ii = gridIndex(i, j)
            if grid[ii][2] > 0:
                grid[ii] /= grid[ii][2]
                grid[ii] += Vec3D(0, -200 * dt, 0)

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

        svd_u, sig, svd_v = svd(F)

        for i in ti.static(range(2 * int(plastic))):
            sig[i + 2*i] = clamp(sig[i+2*i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        oldJ = determinant(F)

        F = mulMat(mulMat(svd_u, sig), transposed(svd_v))

        Jp_new = clamp(particles[index].Jp * oldJ / determinant(F), 0.6, 20.0)
        particles[index].Jp = Jp_new
        particles[index].F = F


def on_press(key):
    try:
        if key.char == 'q':
            init_particles()
    except AttributeError:
        pass

if __name__ == '__main__':
    init_particles()

    frame: tint = 0
    last_frame: tint = frame
    step: tint = 0

    gui = ti.GUI('Taichi MLS-MPM', res=(window_size, window_size))

    while gui.running:
        advance(dt)

        if (step % int(frame_dt / dt)) == 0:
            gui.clear(0x112F41)
            gui.circles(particles.x.to_numpy(), radius=2, color=particles.c.to_numpy())
            gui.rect(topleft = [0.04, 0.04], bottomright=[0.96, 0.96], color=0x068587)
            gui.show()

            if gui.get_event(ti.GUI.SPACE):
                if frame - last_frame > 60:
                    last_frame = frame
                    add_random_particles()

            frame += 1

        step += 1
