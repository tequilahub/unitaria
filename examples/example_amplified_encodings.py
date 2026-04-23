# This file implements the numerical experiment from
# https://arxiv.org/abs/2411.16435

import unitaria as ut
import numpy as np


def g(x):
    H = ut.ConstantUnitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    Hx = H @ x
    return ut.ConstantVector(np.array([1, 1])) - (1 / 4) * ut.ComponentwiseMul(Hx, Hx)


def Dg(x):
    H = ut.ConstantUnitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    return -(1 / 2) * ut.ComponentwiseMul(H @ x) @ H


def Dg_inv(x, gx, simulate):
    Dg_mat = Dg(x)
    print(gx.normalization)
    if simulate:
        norm = gx.simulate_norm()
    else:
        norm = gx.compute_norm()
    Dg_inv_mat = ut.Pseudoinverse(Dg_mat, condition=6, tolerance=0.1 / norm)
    degree = Dg_inv_mat.get_definition().coefficients.degree()
    print(f"Computing inverse with polynomial degree {degree}")
    return Dg_inv_mat @ gx


def amplify(x, simulate):
    if simulate:
        norm = x.simulate_norm()
    else:
        norm = x.compute_norm()

    information_efficiency = norm / x.normalization
    k = 2 * int(np.ceil(0.25 * (np.pi / np.arcsin(information_efficiency) - 2))) + 1
    remove_efficiency = information_efficiency / np.sin(np.pi / (2 * k))

    res = ut.Scale(
        ut.GroverAmplification(ut.Scale(x, remove_efficiency=remove_efficiency), (k - 1) // 2), norm, absolute=True
    )
    return res


def fixed_point(f, x0, n, simulate=False):
    x = x0
    for i in range(n):
        x = amplify(f(x), simulate)

    return x


def newton(f, Df_inv, x0, n, simulate=False):
    x = x0
    for i in range(n):
        fx = f(x)
        Df_inv_fx = Df_inv(x, fx, simulate)
        x = amplify(x - Df_inv_fx, simulate)

    return x


def test_fixed_point():
    simulate = False

    print("Fixed point")
    x0 = ut.ConstantVector(np.array([1, 1]))

    for n in range(4):
        xn = fixed_point(g, x0, n, simulate)
        if simulate:
            solution = xn.simulate()
        else:
            solution = xn.toarray()
        print(f"x{n} = {solution}")


def test_newton():
    simulate = False

    print("Newton")
    x0 = ut.ConstantVector(np.array([2, 0.25]))

    for n in range(3):
        xn = newton(g, Dg_inv, x0, n, simulate)
        if simulate:
            solution = xn.simulate()
        else:
            solution = xn.toarray()
        print(f"x{n} = {solution}")


if __name__ == "__main__":
    test_fixed_point()
    test_newton()
