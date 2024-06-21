import taichi as ti
import numpy as np
from taichialgebra import *

ti.init(arch=ti.gpu, default_fp=ti.f64)

@ti.kernel
def main():
    a2d = Vec2D(3, 5)
    b2d = Vec2D(7, 9)

    addresult2d = add2D(a2d, b2d)
    if (Vec2DEq(addresult2d, Vec2D(10, 14))):
        print("check")
    else:
        print("add2D test failed")

    subresult2d = sub2D(a2d, b2d)
    if (Vec2DEq(subresult2d, Vec2D(-4, -4))):
        print("check")
    else:
        print("sub2D test failed")

    a3d = Vec3D(3, 5, 7)
    b3d = Vec3D(7, 9, 11)

    addresult3d = add3D(a3d, b3d)
    if (Vec3DEq(addresult3d, Vec3D(10, 14, 18))):
        print("check")
    else:
        print("add3D test failed")

    a2x2 = Mat(1, 2, 3, 4)
    determinantresult = determinant(a2x2)
    if (determinantresult == -2):
        print("check")
    else:
        print("determinant test failed")

    transposedresult = transposed(a2x2)
    if (Mateq(transposedresult, Mat(1, 3, 2, 4))):
        print("check")
    else:
        print("transposed test failed")

    b2x2 = Mat(5, 6, 7, 8)
    mulMatresult = mulMat(a2x2, b2x2)
    if (Mateq(mulMatresult, Mat(19, 22, 43, 50))):
        print("check")
    else:
        print("mulMat test failed")

    mulMatVecresult = mulMatVec(a2x2, a2d)
    if (Vec2DEq(mulMatVecresult, Vec2D(18, 26))):
        print("check")
    else:
        print("mulMatVec test failed")

    addmatresult = addMat(a2x2, b2x2)
    if (Mateq(addmatresult, Mat(6, 8, 10, 12))):
        print("check")
    else:
        print("addMat test failed")

    submatresult = subMat(a2x2, b2x2)
    if (Mateq(submatresult, Mat(-4, -4, -4, -4))):
        print("check")
    else:
        print("subMat test failed")

    outerproductresult = outer_product(a2d, b2d)
    if (Mateq(outerproductresult, Mat(21, 35, 27, 45))):
        print("check")
    else:
        print("outer_product test failed")

    clampresult = clamp(15, 0, 10)
    if (clampresult == 10):
        print("check")
    else:
        print("clamp test failed")

    clampresult2 = clamp(5, 0, 10)
    if (clampresult2 == 5):
        print("check")
    else:
        print("clamp test failed")

    clampresult3 = clamp(-5, 0, 10)
    if (clampresult3 == 0):
        print("check")
    else:
        print("clamp test failed")

    rresult, sresult = polar_decomp(a2x2)
    if (Mateq(rresult, Mat(0.980580675691, 0.196116135138, -0.196116135138, 0.980580675691))):
        print("check")
    else:
        print("polar_decomp test failed")
        print(rresult)
        print(Mat(0.980580675691, 0.196116135138, -0.196116135138, 0.980580675691))

    if (Mateq(sresult, Mat(0.5883484054146, 2.15727748652, 2.15727748652, 4.510671108178))):
        print("check")
    else:
        print("polar_decomp test failed")
        print(sresult)
        print(Mat(0.5883484054146, 2.15727748652, 2.15727748652, 4.510671108178))

    uresult, sresult, vresult = svd(a2x2)
    if (Mateq(uresult, Mat(-0.5760484367663, 0.8174155604704, -0.8174155604704, -0.5760484367663))):
        print("check")
    else:
        print("svd test failed")
        print(uresult)
        print(Mat(-0.5760484367663, 0.8174155604704, -0.8174155604704, -0.5760484367663))

    if (Mateq(sresult, Mat(5.464985704219, 0.0, 0.0, -0.3659661906263))):
        print("check")
    else:
        print("svd test failed")
        print(sresult)
        print(Mat(5.464985704219, 0.0, 0.0, -0.3659661906263))

    if (Mateq(vresult, Mat(-0.4045535848338, 0.9145142956773, -0.9145142956773, -0.4045535848338))):
        print("check")
    else:
        print("svd test failed")
        print(vresult)
        print(Mat(-0.4045535848338, 0.9145142956773, -0.9145142956773, -0.4045535848338))

    hadamardresult = had2D(a2d, b2d)
    if (Vec2DEq(hadamardresult, Vec2D(21, 45))):
        print("check")
    else:
        print("had2D test failed")

main()

