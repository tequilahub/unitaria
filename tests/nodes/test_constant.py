import numpy as np
import scipy.stats
import unitaria as ut


def test_constant_vector():
    A = ut.ConstantVector(np.array([1, 2j, 1 / 3, -1j / 4]))
    ut.verify(A)
    ut.verify(ut.ConstantUnitary(np.sqrt(1 / 2) * np.array([[1, 1], [1, -1]])))
    ut.verify(ut.ConstantUnitary(np.array([[0, 1], [1, 0]])))
    angle = 1.23
    ut.verify(ut.ConstantUnitary(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])))
    ut.verify(ut.ConstantUnitary(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])))


def test_constant_unitary():
    ut.verify(ut.ConstantUnitary(np.eye(1)))
    ut.verify(ut.ConstantUnitary(np.eye(2)))
    ut.verify(ut.ConstantUnitary(np.eye(4)))

    for i in range(1, 5):
        U = scipy.stats.unitary_group.rvs(2**i, random_state=0)
        ut.verify(ut.ConstantUnitary(U))

    # Triggers an edge case because of the degenerate eigenvalue
    U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, -0.81649658, -0.57735027, 0], [0, -0.57735027, 0.81649658, 0]])
    ut.verify(ut.ConstantUnitary(U))


def test_constant_unitary_rectangular():
    ut.verify(ut.ConstantUnitary(np.array([[1, 0]])))
    ut.verify(ut.ConstantUnitary(np.array([[1], [0]])))
    ut.verify(ut.ConstantUnitary(np.array([[0], [1]])))
    ut.verify(ut.ConstantUnitary(np.array([[np.sqrt(1 / 2)], [np.sqrt(1 / 2)]])))


def test_global_phase():
    A = ut.ConstantVector(np.array([1, -1]))
    ut.verify(A)
    A = ut.ConstantVector(np.array([-1, 1]))
    ut.verify(A)
