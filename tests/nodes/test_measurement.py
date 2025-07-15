from bequem.nodes.constant import ConstantUnitary
import numpy as np
import pytest


identity = ConstantUnitary(np.eye(2))
z_pauli_matrix = ConstantUnitary(np.array([[1, 0], [0, -1]]))

def test_sum_1_norm():
    sum_1 = 0.75 * identity + 0.25 * z_pauli_matrix
    sum_1_expected_result = ConstantUnitary(np.array([[1, 0], [0, 0.5]]))
    assert pytest.approx(sum_1.simulate_norm()) == sum_1_expected_result.simulate_norm()
    assert pytest.approx(sum_1.simulate_norm()) == 1.0

def test_sum_2_norm():
    sum_2 = 0.5 * identity + 0.5 * z_pauli_matrix
    sum_2_expected_result = ConstantUnitary(np.array([[1, 0], [0, 0]]))
    assert pytest.approx(sum_2.simulate_norm()) == sum_2_expected_result.simulate_norm() 
    assert pytest.approx(sum_2.simulate_norm()) == 1.0 
