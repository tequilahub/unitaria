import numpy as np

from unitaria.nodes.constants.constant_unitary import ConstantUnitary
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.verifier import verify


def test_constant_vector():
    A = ConstantVector(np.array([1, 2j, 1 / 3, -1j / 4]))
    verify(A)
    verify(ConstantUnitary(np.sqrt(1 / 2) * np.array([[1, 1], [1, -1]])))
    verify(ConstantUnitary(np.array([[0, 1], [1, 0]])))
    angle = 1.23
    verify(ConstantUnitary(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])))
    verify(ConstantUnitary(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])))


def test_constant_unitary():
    verify(ConstantUnitary(np.eye(1)))
    verify(ConstantUnitary(np.eye(2)))
    verify(ConstantUnitary(np.eye(4)))


def test_constant_unitary_rectangular():
    verify(ConstantUnitary(np.array([[1, 0]])))
    verify(ConstantUnitary(np.array([[1], [0]])))
    verify(ConstantUnitary(np.array([[0], [1]])))
    verify(ConstantUnitary(np.array([[np.sqrt(1 / 2)], [np.sqrt(1 / 2)]])))


def test_global_phase():
    A = ConstantVector(np.array([1, -1]))
    verify(A)
    A = ConstantVector(np.array([-1, 1]))
    verify(A)
