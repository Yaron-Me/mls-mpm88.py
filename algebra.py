import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.13}".format(x)})

def add2D(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[0] + b[0], a[1] + b[1]])

def sub2D(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[0] - b[0], a[1] - b[1]])

def add3D(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[0] + b[0], a[1] + b[1], a[2] + b[2]])

def determinant(a : np.ndarray) -> float:
    return a[0] * a[3] - a[1] * a[2]

def transposed(a: np.ndarray) -> np.ndarray:
    return np.array([a[0], a[2], a[1], a[3]])

def mulMat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[0]*b[0]+a[1]*b[2],
        a[0]*b[1]+a[1]*b[3],
        a[2]*b[0]+a[3]*b[2],
        a[2]*b[1]+a[3]*b[3]
    ])

def mulMatVec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[0]*b[0]+a[2]*b[1],
        a[1]*b[0]+a[3]*b[1]
    ])

def addMat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[0]+b[0], a[1]+b[1],
        a[2]+b[2], a[3]+b[3]
    ])

def subMat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[0]-b[0], a[1]-b[1],
        a[2]-b[2], a[3]-b[3]
    ])

def outer_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[0]*b[0],a[1]*b[0],
        a[0]*b[1],a[1]*b[1]
    ])

def clamp(x: float, min_: float, max_: float) -> float:
    return min(max(x, min_), max_)

def polar_decomp(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x: float = m[0] + m[3]
    y: float = m[2] - m[1]
    scale: float = 1.0 / np.sqrt(x*x + y*y)
    c: float = x * scale
    s: float = y * scale
    R: np.ndarray = np.array([c, s, -s, c])

    S: np.ndarray = mulMat(m, R)

    return R, S

def svd(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, S = polar_decomp(m)
    c: float
    s: float
    sig: np.ndarray
    V: np.ndarray
    if (abs(S[1]) < 1e-6):
        sig = S.copy()
        c = 1
        s = 0
    else:
        tao: float = 0.5 * (S[0] - S[3])
        w: float = np.sqrt(tao*tao + S[1]*S[1])
        t: float = S[1] / (tao + w) if tao > 0 else S[1] / (tao - w)
        c = 1.0 / np.sqrt(t*t +1)
        s = -t * c
        sig = np.array([0.0, 0.0, 0.0, 0.0])
        sig[0] = c * c * S[0] - 2 * c * s * S[1] + s * s * S[3]
        sig[3] = s * s * S[0] + 2 * c * s * S[1] + c * c * S[3]
    if (sig[0] < sig[3]):
        tmp = sig[0]
        sig[0] = sig[3]
        sig[3] = tmp
        V = np.array([-s, -c, c, -s])
    else:
        V = np.array([c, -s, s, c])

    V = transposed(V)
    U = mulMat(U, V)

    return U, sig, V

def had2D(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([a[0]*b[0], a[1]*b[1]])