import taichi as ti

Vec2D = ti.math.vec2
Vec3D = ti.math.vec3
Mat = ti.math.vec4
tfloat = ti.types.f64
tint = ti.types.i64

@ti.func
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

@ti.func
def Vec2DEq(a: Vec2D, b: Vec2D) -> bool:
    fail = False
    for i in ti.static(range(2)):
        if not isclose(a[i], b[i]):
            fail = True
    return not fail

@ti.func
def Vec3DEq(a: Vec3D, b: Vec3D) -> bool:
    fail = False
    for i in ti.static(range(3)):
        if not isclose(a[i], b[i]):
            fail = True
    return not fail

@ti.func
def Mateq(a: Mat, b: Mat) -> bool:
    fail = False
    for i in ti.static(range(4)):
        if not isclose(a[i], b[i]):
            fail = True
    return not fail

@ti.func
def add2D(a: Vec2D, b: Vec2D) -> Vec2D:
    return Vec2D(a[0] + b[0], a[1] + b[1])

@ti.func
def sub2D(a: Vec2D, b: Vec2D) -> Vec2D:
    return Vec2D(a[0] - b[0], a[1] - b[1])

@ti.func
def add3D(a: Vec3D, b: Vec3D) -> Vec3D:
    return Vec3D(a[0] + b[0], a[1] + b[1], a[2] + b[2])

@ti.func
def determinant(a : Mat) -> tfloat:
    return a[0] * a[3] - a[1] * a[2]

@ti.func
def transposed(a: Mat) -> Mat:
    return Mat(a[0], a[2], a[1], a[3])

@ti.func
def mulMat(a: Mat, b: Mat) -> Mat:
    return Mat(
        a[0]*b[0]+a[1]*b[2],
        a[0]*b[1]+a[1]*b[3],
        a[2]*b[0]+a[3]*b[2],
        a[2]*b[1]+a[3]*b[3]
    )

@ti.func
def mulMatVec(a: Mat, b: Vec2D) -> Vec2D:
    return Vec2D(
        a[0]*b[0]+a[2]*b[1],
        a[1]*b[0]+a[3]*b[1]
    )

@ti.func
def addMat(a: Mat, b: Mat) -> Mat:
    return Mat(
        a[0]+b[0], a[1]+b[1],
        a[2]+b[2], a[3]+b[3]
    )

@ti.func
def subMat(a: Mat, b: Mat) -> Mat:
    return Mat(
        a[0]-b[0], a[1]-b[1],
        a[2]-b[2], a[3]-b[3]
    )

@ti.func
def outer_product(a: Vec2D, b: Vec2D) -> Mat:
    return Mat(
        a[0]*b[0],a[1]*b[0],
        a[0]*b[1],a[1]*b[1]
    )

@ti.func
def clamp(x: tfloat, min_: tfloat, max_: tfloat) -> tfloat:
    return min(max(x, min_), max_)

@ti.func
def polar_decomp(m: Mat) -> tuple[Mat, Mat]:
    x = m[0] + m[3]
    y = m[2] - m[1]
    scale = 1.0 / ti.math.sqrt(x*x + y*y)
    c = x * scale
    s = y * scale

    R = Mat(c, s, -s, c)

    S = mulMat(m, R)

    return R, S

@ti.func
def svd(m: Mat) -> tuple[Mat, Mat, Mat]:
    U, S = polar_decomp(m)
    c: tfloat = 0.0
    s: tfloat = 0.0
    sig = Mat(0.0, 0.0, 0.0, 0.0)
    V = Mat(0.0, 0.0, 0.0, 0.0)
    if (abs(S[1]) < 1e-6):
        sig = S
        c = 1.0
        s = 0.0
    else:
        tao: tfloat = 0.5 * (S[0] - S[3])
        w: tfloat = ti.math.sqrt(tao*tao + S[1]*S[1])
        t: tfloat = S[1] / (tao + w) if tao > 0 else S[1] / (tao - w)
        c = 1.0 / ti.math.sqrt(t*t +1)
        s = -t * c
        sig = Mat(0.0, 0.0, 0.0, 0.0)
        sig[0] = c * c * S[0] - 2 * c * s * S[1] + s * s * S[3]
        sig[3] = s * s * S[0] + 2 * c * s * S[1] + c * c * S[3]
    if (sig[0] < sig[3]):
        tmp = sig[0]
        sig[0] = sig[3]
        sig[3] = tmp
        V = Mat(-s, -c, c, -s)
    else:
        V = Mat(c, -s, s, c)

    V = transposed(V)
    U = mulMat(U, V)

    return U, sig, V

@ti.func
def had2D(a: Vec2D, b: Vec2D) -> Vec2D:
    return Vec2D(a[0]*b[0], a[1]*b[1])