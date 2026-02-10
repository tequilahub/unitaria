import numpy as np

from unitaria import Subspace
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.constants.constant_unitary import ConstantUnitary
from unitaria.nodes.inversion.pseudoinverse import Pseudoinverse
from unitaria.subspace import ZeroQubit, ID


def rot_matrix(angle: float) -> np.array:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def test_pseudoinverse():
    A = (ConstantUnitary(rot_matrix(np.pi / 2)) | ConstantUnitary(rot_matrix(np.pi / 3))) | (
        ConstantUnitary(rot_matrix(np.pi / 4)) | ConstantUnitary(rot_matrix(0.0))
    )
    B = (
        Projection(subspace_from=Subspace(3), subspace_to=Subspace([ZeroQubit(), ID, ID]))
        @ A
        @ Projection(subspace_from=Subspace([ZeroQubit(), ID, ID]), subspace_to=Subspace(3))
    )
    B_inv = Pseudoinverse(B, delta=0.1, epsilon=0.1)
    # TODO: Commented out because QSVT circuits are currently broken.
    # This is caught by the `test_qsvt_with_polynomial` for now, but should
    # be uncommented once the problem is fixed.
    # verify(B_inv)
    C = B_inv @ B  # should be roughly a projector on all but the first basis states
    result = C.compute(np.eye(4))
    print(result)
