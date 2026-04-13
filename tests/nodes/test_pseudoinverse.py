import numpy as np
import unitaria as ut


def rot_matrix(angle: float) -> np.array:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def test_pseudoinverse():
    A = (ut.ConstantUnitary(rot_matrix(np.pi / 2)) | ut.ConstantUnitary(rot_matrix(np.pi / 3))) | (
        ut.ConstantUnitary(rot_matrix(np.pi / 4)) | ut.ConstantUnitary(rot_matrix(0.0))
    )
    B = (
        ut.Projection(subspace_from=ut.Subspace(bits=3), subspace_to=ut.Subspace([ut.ZeroQubit(), ut.ID, ut.ID]))
        @ A
        @ ut.Projection(subspace_from=ut.Subspace([ut.ZeroQubit(), ut.ID, ut.ID]), subspace_to=ut.Subspace(bits=3))
    )
    B_inv = ut.Pseudoinverse(B, condition=10.0, tolerance=0.1)
    C = B_inv @ B  # should be roughly a projector on all but the first basis states
    ut.verify(C, np.diag(np.array([0.0, 1.0, 1.0, 1.0])), atol=0.1)


def test_pseudoinverse_normalization():
    A = 2 * ut.ConstantUnitary(rot_matrix(np.pi / 3))
    B = (
        ut.Projection(subspace_from=ut.Subspace(bits=1), subspace_to=ut.Subspace([ut.ZeroQubit()]))
        @ A
        @ ut.Projection(subspace_from=ut.Subspace([ut.ZeroQubit()]), subspace_to=ut.Subspace(bits=1))
    )
    B_inv = ut.Pseudoinverse(B, condition=2.0, tolerance=0.1)
    C = B_inv @ B
    ut.verify(C, np.array([[1.0]]), atol=0.1)
