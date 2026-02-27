import numpy as np

from unitaria import Subspace
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.constants.constant_unitary import ConstantUnitary
from unitaria.nodes.inversion.pseudoinverse import Pseudoinverse
from unitaria.subspace import ZeroQubit, ID
from unitaria.verifier import verify


def rot_matrix(angle: float) -> np.array:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def test_pseudoinverse():
    A = (ConstantUnitary(rot_matrix(np.pi / 2)) | ConstantUnitary(rot_matrix(np.pi / 3))) | (
        ConstantUnitary(rot_matrix(np.pi / 4)) | ConstantUnitary(rot_matrix(0.0))
    )
    B = (
        Projection(subspace_from=Subspace(bits=3), subspace_to=Subspace([ZeroQubit(), ID, ID]))
        @ A
        @ Projection(subspace_from=Subspace([ZeroQubit(), ID, ID]), subspace_to=Subspace(bits=3))
    )
    B_inv = Pseudoinverse(B, condition=10.0, tolerance=0.1)
    C = B_inv @ B  # should be roughly a projector on all but the first basis states
    verify(C, np.diag(np.array([0.0, 1.0, 1.0, 1.0])), atol=0.1)


def test_pseudoinverse_normalization():
    A = 2 * ConstantUnitary(rot_matrix(np.pi / 3))
    B = (
        Projection(subspace_from=Subspace(bits=1), subspace_to=Subspace([ZeroQubit()]))
        @ A
        @ Projection(subspace_from=Subspace([ZeroQubit()]), subspace_to=Subspace(bits=1))
    )
    B_inv = Pseudoinverse(B, condition=2.0, tolerance=0.1)
    C = B_inv @ B
    verify(C, np.array([[1.0]]), atol=0.1)
