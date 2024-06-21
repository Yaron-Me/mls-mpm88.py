from algebra import *
import numpy as np
import cv2

def arr(array):
    return np.array(array)

class Particle:
    def __init__(self, x: np.ndarray, c: int, v: np.ndarray = arr([0, 0])):
        self.x = x
        self.v = v
        self.F = arr([1, 0, 0, 1])
        self.C = arr([0, 0, 0, 0])
        self.Jp = 1
        self.c = c

# Window
window_size: int = 800

# Grid resolution
n: int = 80

dt: float = 1e-4
frame_dt: float = 1e-3
dx: float = 1.0 / n
inv_dx: float = 1.0 / dx

# Snow material properties
particle_mass: float = 1.0
vol: float = 1.0
hardening: float = 10.0
E: float = 1e4
nu: float = 0.2
plastic: bool = True

# initial Lame parameters
mu_0: float = E / (2 * (1 + nu))
lambda_0: float = E * nu / ((1 + nu) * (1 - 2 * nu))

particles: list[Particle] = []

def gridIndex(i, j):
    return i + (n + 1) * j

def advance(dt: float):
    # reset grid
    grid = np.zeros(((n + 1) * (n + 1), 3))

    # P2G
    for p in particles:
        base_coord = sub2D((p.x * inv_dx), arr([0.5,0.5])).astype(int)
        fx = sub2D((p.x * inv_dx), base_coord)

        w = [
            had2D(arr([0.5, 0.5]), np.square(sub2D(arr([1.5, 1.5]), fx))),
            sub2D(arr([0.75, 0.75]), np.square(sub2D(fx, arr([1.0, 1.0])))),
            had2D(arr([0.5, 0.5]),np.square( sub2D(fx, arr([0.5, 0.5]))))
        ]

        e = np.exp(hardening * (1.0 - p.Jp))

        mu = mu_0 * e
        lambda_ = lambda_0 * e

        J = determinant(p.F)

        r, s = polar_decomp(p.F)

        k1 = -4*inv_dx*inv_dx*dt*vol

        k2 = lambda_*(J-1)*J

        stress = addMat(mulMat(subMat(transposed(p.F),r),p.F) * 2 * mu, arr([k2,0,0,k2])) * k1
        affine = addMat(stress, p.C * particle_mass)

        mv = arr([p.v[0] * particle_mass, p.v[1] * particle_mass, particle_mass])
        for i in range(3):
            for j in range(3):
                dpos = arr([(i-fx[0])*dx, (j-fx[1])*dx])
                weight = w[i][0] * w[j][1]
                ii = gridIndex(base_coord[0] + i, base_coord[1] + j)
                grid[ii] = add3D(grid[ii], add3D(mv, np.hstack((mulMatVec(affine, dpos), 0))) * weight)

    # modify grid velocities to respect boundaries
    boundary: float = 0.05
    for i in range(n + 1):
        for j in range(n + 1):
            ii = gridIndex(i, j)
            if (grid[ii][2] > 0):
                grid[ii] = grid[ii] / grid[ii][2]
                grid[ii] = add3D(grid[ii], arr([0, -200*dt, 0]))

                x: float = i / n
                y: float = j / n

                if (x < boundary or x > 1-boundary or y > 1-boundary):
                    grid[ii] = arr([0, 0, 0])

                if (y < boundary):
                    grid[ii][1] = max(0.0, grid[ii][1])

    # G2P
    for p in particles:
        base_coord = sub2D(p.x * inv_dx, arr([0.5, 0.5])).astype(int)
        fx = sub2D(p.x * inv_dx, base_coord)

        w = [
            had2D(arr([0.5, 0.5]), np.square(sub2D(arr([1.5, 1.5]), fx))),
            sub2D(arr([0.75, 0.75]), np.square(sub2D(fx, arr([1.0, 1.0])))),
            had2D(arr([0.5, 0.5]),np.square( sub2D(fx, arr([0.5, 0.5]))))
        ]

        p.C = arr([0, 0, 0, 0])
        p.v = arr([0, 0])

        for i in range(3):
            for j in range(3):
                dpos = sub2D(arr([i, j]), fx)
                weight = w[i][0] * w[j][1]

                ii = gridIndex(base_coord[0] + i, base_coord[1] + j)

                p.v = add2D(p.v, grid[ii] * weight)

                p.C = addMat(p.C, outer_product(grid[ii] * weight, dpos) * (4 * inv_dx))

        # advection
        p.x = add2D(p.x, p.v * dt)

        F = mulMat(p.F, addMat(arr([1, 0, 0, 1]), p.C * dt))

        svd_u, sig, svd_v = svd(F)

        for i in range(2 * int(plastic)):
            sig[i + 2*i] = clamp(sig[i+2*i], 1.0 - 2.5e-2, 1.0 + 7.5e-3)

        oldJ = determinant(F)

        F = mulMat(mulMat(svd_u, sig), transposed(svd_v))

        Jp_new = clamp(p.Jp * oldJ / determinant(F), 0.6, 20.0)
        p.Jp = Jp_new
        p.F = F


def add_object(center: np.ndarray, c: int):
    for i in range(500):
        # generate random x between 0 and 1
        x = np.random.rand()
        y = np.random.rand()
        coord = (arr([x, y]) * 2 - arr([1, 1])) * 0.08 + center
        particles.append(Particle(coord, c))


def draw(step):
    img = np.zeros((window_size, window_size, 3), dtype=np.uint8)

    # creat img filled with 0x112F41
    img[:, :] = (0x41, 0x2F, 0x11)

    for p in particles:
        x = int(p.x[0] * window_size)
        y = int(p.x[1] * window_size)

        # convert color from int to tuple
        color = tuple(int.to_bytes(p.c, 4, 'little'))

        cv2.circle(img, (x, y), 1, color, -1)

    x1 = int(0.04 * window_size)
    y1 = int(0.04 * window_size)
    x2 = int(0.96 * window_size)
    y2 = int(0.96 * window_size)

    color = tuple(int.to_bytes(0x4FB99F, 4, 'little'))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # image needs to be flipped on y-axis
    img = cv2.flip(img, 0)

    cv2.imshow('image', img)
    # cv2.imwrite('step: ' + str(step) + '.jpeg', img)
    cv2.waitKey(1)

if __name__ == '__main__':

    add_object(arr([0.55,0.45]), 0xED553B)
    add_object(arr([0.45,0.65]), 0xF2B134)
    add_object(arr([0.55,0.85]), 0x068587)

    # particles.append(Particle(arr([0.55, 0.45]), 0xED553B))

    frame: int = 0
    step: int = 0
    while True:
        advance(dt)
        # exit()

        if (step % int(frame_dt / dt)) == 0:
            draw(step)

        step += 1