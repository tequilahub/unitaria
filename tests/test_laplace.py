import pytest

import numpy as np

from bequem.nodes import Increment, Identity, Add, Scale, Adjoint, ConstantUnitary
from bequem.subspace import Subspace


@pytest.mark.parametrize("n", range(1, 5))
def test_laplace(n: int):
    C = Increment(n)
    A = Add(Scale(Identity(Subspace(n)), -2), Scale(Add(C, Adjoint(C)), 1))
    A.verify()


def test_preconditioned_laplace_1d():
    L = 3

    T = Identity(Subspace(L))

    for l in range(L, 0, -1):
        I_l = Identity(Subspace.from_dim(2**l - 1, bits=l), Subspace(l))
        N_l = Increment(l) @ I_l
        C_l = 2**(l / 2) * (I_l - N_l)
        C_l.verify()

        if l != L:
            T_l = ConstantUnitary(
                np.sqrt(1 / 2) * np.array([
                    [1, -np.sqrt(3) / 2],
                    [0, 1 / 2],
                    [1, np.sqrt(3) / 2],
                    [0, 1 / 2],
                ])) & Identity(Subspace(l - 1))
            T_l.verify(up_to_phase=True)
            T = T @ T_l

        T.verify(up_to_phase=True)

        print(f"C_{l}")
        print(C_l.circuit.draw())
        print("T")
        print(T.circuit.draw())

        print(f"T @ C_{l}")
        C_l = 2**(-l * (2 - 1) / 2) * T @ C_l
        print(C_l.circuit.draw())

        C_l.verify(up_to_phase=True)
