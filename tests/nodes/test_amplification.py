import numpy as np

from unitaria import Subspace
from unitaria.nodes.amplification.fixed_point_amplification import FixedPointAmplification
from unitaria.nodes.amplification.linear_amplification import LinearAmplification
from unitaria.nodes.basic.mul import Mul
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.constants.constant_unitary import ConstantUnitary
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.amplification.grover_amplification import GroverAmplification
from unitaria.subspace import ZeroQubit, ID


def test_grover_amplification():
    node = ConstantVector(np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]))
    proj = Projection(subspace_from=Subspace(bits=2, zero_qubits=0), subspace_to=Subspace(bits=0, zero_qubits=2))
    node = Mul(node, proj)

    # Test amplifying to 1
    amplified = GroverAmplification(node, 1)
    assert np.isclose(amplified.compute_norm(np.array([1])), 1)

    # Test "overshooting" back down to 0.5
    amplified = GroverAmplification(node, 2)
    assert np.isclose(amplified.compute_norm(np.array([1])), 0.5)


def test_fixed_point_amplification():
    node = ConstantVector(np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]))
    proj = Projection(subspace_from=Subspace(bits=2, zero_qubits=0), subspace_to=Subspace(bits=0, zero_qubits=2))
    node = Mul(node, proj)

    # Test with known norm
    amplified = FixedPointAmplification(node, 0.5, 0.1)
    assert amplified.compute_norm(np.array([1])) > 1 - 0.1

    # Test with unknown (but lower-bounded) norm
    amplified = FixedPointAmplification(node, 0.1, 0.1)
    assert amplified.compute_norm(np.array([1])) > 1 - 0.1


def rot_matrix(angle: float) -> np.array:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def test_linear_amplification():
    A = (ConstantUnitary(rot_matrix(np.arccos(0.0))) | ConstantUnitary(rot_matrix(np.arccos(0.1)))) | (
        ConstantUnitary(rot_matrix(np.arccos(0.2))) | ConstantUnitary(rot_matrix(np.arccos(0.3)))
    )
    node = (
        Projection(subspace_from=Subspace(bits=3), subspace_to=Subspace(registers=[ZeroQubit(), ID, ID]))
        @ A
        @ Projection(subspace_from=Subspace(registers=[ZeroQubit(), ID, ID]), subspace_to=Subspace(bits=3))
    )
    amplified = LinearAmplification(node, 2.0, 0.4, 0.1)
    result = amplified.compute(np.eye(4))
    assert np.allclose(result, np.diag(np.array([0.0, 0.2, 0.4, 0.6])), atol=0.1)
