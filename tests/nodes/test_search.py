import numpy as np

from unitaria import Subspace
from unitaria.nodes.basic.mul import Mul
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.search.amplitude_amplification import AmplitudeAmplification


def test_amplitude_amplification():
    node = ConstantVector(np.array([1 / 2, 1 / 2, 1 / 2, 1 / 2]))
    proj = Projection(subspace_from=Subspace(2, 0), subspace_to=Subspace(0, 2))
    node = Mul(node, proj)
    amplified = AmplitudeAmplification(node, 1)
    assert np.isclose(amplified.compute_norm(np.array([1])), 1)
