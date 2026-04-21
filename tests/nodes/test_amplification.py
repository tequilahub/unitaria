import numpy as np
import unitaria as ut


def test_grover_amplification():
    node = ut.ConstantVector(np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]))
    proj = ut.Projection(subspace_from=ut.Subspace("##"), subspace_to=ut.Subspace("00"))
    node = proj @ node

    amplified = ut.GroverAmplification(node, 0)
    ut.verify(amplified, np.array([0.5]))

    # Test amplifying to 1
    amplified = ut.GroverAmplification(node, 1)
    ut.verify(amplified, np.array([1]))

    # Test "overshooting" back down to 0.5
    amplified = ut.GroverAmplification(node, 2)
    ut.verify(amplified, np.array([0.5]))


def test_fixed_point_amplification():
    node = ut.ConstantVector(np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]))
    proj = ut.Projection(subspace_from=ut.Subspace("##"), subspace_to=ut.Subspace("00"))
    node = proj @ node

    # Test with known norm
    amplified = ut.FixedPointAmplification(node, 0.5, 0.1)
    ut.verify(amplified)
    assert amplified.compute_norm(np.array([1])) > 1 - 0.1

    # Test with unknown (but lower-bounded) norm
    amplified = ut.FixedPointAmplification(node, 0.1, 0.1)
    ut.verify(amplified)
    assert amplified.compute_norm(np.array([1])) > 1 - 0.1


def rot_matrix(angle: float) -> np.array:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def test_linear_amplification():
    A = (ut.ConstantUnitary(rot_matrix(np.arccos(0.0))) | ut.ConstantUnitary(rot_matrix(np.arccos(0.1)))) | (
        ut.ConstantUnitary(rot_matrix(np.arccos(0.2))) | ut.ConstantUnitary(rot_matrix(np.arccos(0.3)))
    )
    node = (
        ut.Projection(subspace_from=ut.Subspace("###"), subspace_to=ut.Subspace("##0"))
        @ A
        @ ut.Projection(subspace_from=ut.Subspace("##0"), subspace_to=ut.Subspace("###"))
    )
    amplified = ut.LinearAmplification(node, 2.0, 0.4, 0.1)
    ut.verify(amplified, np.diag(np.array([0.0, 0.2, 0.4, 0.6])), atol=0.1, check_adjoint=False)
