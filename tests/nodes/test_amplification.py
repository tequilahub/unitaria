import numpy as np

from unitaria import Subspace
from unitaria.nodes.amplification.fixed_point_amplification import FixedPointAmplification
from unitaria.nodes.amplification.singular_value_amplification import SingularValueAmplification
from unitaria.nodes.basic.mul import Mul
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.classical.increment import Increment
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.amplification.grover_amplification import GroverAmplification


def test_grover_amplification():
    node = ConstantVector(np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]))
    proj = Projection(subspace_from=Subspace(2, 0), subspace_to=Subspace(0, 2))
    node = Mul(node, proj)
    amplified = GroverAmplification(node, 1)
    assert np.isclose(amplified.compute_norm(np.array([1])), 1)


def test_fixed_point_amplification():
    node = ConstantVector(np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]))
    proj = Projection(subspace_from=Subspace(2, 0), subspace_to=Subspace(0, 2))
    node = Mul(node, proj)

    # Test with known norm
    amplified = FixedPointAmplification(node, 0.5, 0.1)
    assert amplified.compute_norm(np.array([1])) > 1 - 0.1

    # Test with unknown (but lower-bounded) norm
    amplified = FixedPointAmplification(node, 0.1, 0.1)
    assert amplified.compute_norm(np.array([1])) > 1 - 0.1


def test_singular_value_amplification():
    inefficiency = Mul(
        ConstantVector(np.array([0.45, np.sqrt(1 - 0.45**2)])),
        Projection(subspace_from=Subspace(1, 0), subspace_to=Subspace(0, 1)),
    )
    node = Increment(3) & inefficiency
    amplified = SingularValueAmplification(node, 2.0, 0.1, 0.1)
    norm0 = amplified.compute_norm(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    norm4 = amplified.compute_norm(np.array([0, 0, 0, 1, 0, 0, 0, 0]))

    assert norm0 > 1 - 0.1 - 0.1
    assert norm4 > 1 - 0.1 - 0.1
    assert np.abs(norm0 - norm4) < 0.1
