import unitaria as ut
import numpy as np
import pytest


def test_sum_1_norm():
    identity = ut.Identity(ut.Subspace("#"))
    z_pauli_matrix = ut.ConstantUnitary(np.array([[1, 0], [0, -1]]))
    sum_1 = 0.75 * identity + 0.25 * z_pauli_matrix
    assert pytest.approx(sum_1.simulate_norm(input=0)) == 1.0
    assert pytest.approx(sum_1.simulate_norm(input=1)) == 0.5


def test_sum_2_norm():
    identity = ut.Identity(ut.Subspace("#"))
    z_pauli_matrix = ut.ConstantUnitary(np.array([[1, 0], [0, -1]]))
    sum_2 = 0.5 * identity + 0.5 * z_pauli_matrix
    assert pytest.approx(sum_2.simulate_norm(input=0)) == 1.0
    assert pytest.approx(sum_2.simulate_norm(input=1)) == 0.0
