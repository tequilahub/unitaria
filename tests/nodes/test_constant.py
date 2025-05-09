import pytest

import numpy as np
from bequem.nodes.constant import ConstantVector, ConstantUnitary


def test_constant_vector():
    A = ConstantVector(np.array([1, 2j, 1 / 3, -1j / 4]))
    A.verify()


def test_constant_unitary():
    ConstantUnitary(np.eye(1)).verify()
    ConstantUnitary(np.eye(2)).verify()
    ConstantUnitary(np.eye(4)).verify()


def test_constant_unitary_rectangular():
    ConstantUnitary(np.array([[1, 0]])).verify()
    ConstantUnitary(np.array([[1], [0]])).verify()
def test_constant_unitary_failing():
    # Global phase is not correct
    ConstantUnitary(np.array([[0, 1], [1, 0]])).verify(up_to_phase=True)
    angle = 1.23
    ConstantUnitary(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])).verify(up_to_phase=True)
    ConstantUnitary(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])).verify(up_to_phase=True)
    ConstantUnitary(np.array([[0], [1]])).verify(up_to_phase=True)
    ConstantUnitary(np.array([[np.sqrt(1/2)], [np.sqrt(1/2)]])).verify(up_to_phase=True)


def test_global_phase():
    A = ConstantVector(np.array([1, -1]))
    A.verify()
    A = ConstantVector(np.array([-1, 1]))
    A.verify()
