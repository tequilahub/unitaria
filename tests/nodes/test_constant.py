import numpy as np
import scipy.stats
import unitaria as ut


def test_constant_vector():
    ut.verify(ut.ConstantVector(np.array([1, 2j, 1 / 3, -1j / 4])))
    ut.verify(ut.ConstantVector(np.array([1, 2j, 1 / 3])))
    ut.verify(ut.ConstantVector(np.array([1])))
    ut.verify(ut.ConstantVector(np.array([0, 0])))


def test_constant_matrix():
    rng = np.random.default_rng(0)
    for i in range(1, 5):
        n, m = rng.integers(1, 2**i, size=2)
        mat = rng.standard_normal((n, m))
        ut.verify(ut.ConstantMatrix(mat), mat)


def test_constant_unitary():
    ut.verify(ut.ConstantUnitary(np.eye(1)))
    ut.verify(ut.ConstantUnitary(np.eye(2)))
    ut.verify(ut.ConstantUnitary(np.eye(4)))

    ut.verify(ut.ConstantUnitary(np.sqrt(1 / 2) * np.array([[1, 1], [1, -1]])))
    ut.verify(ut.ConstantUnitary(np.array([[0, 1], [1, 0]])))
    angle = 1.23
    ut.verify(ut.ConstantUnitary(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])))
    ut.verify(ut.ConstantUnitary(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])))

    rng = np.random.default_rng(0)
    for i in range(1, 5):
        U = scipy.stats.unitary_group.rvs(2**i, random_state=rng)
        ut.verify(ut.ConstantUnitary(U))

    # Triggers an edge case because of the degenerate eigenvalue
    U = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, -0.81649658, -0.57735027, 0], [0, -0.57735027, 0.81649658, 0]])
    ut.verify(ut.ConstantUnitary(U))
    ut.verify(
        ut.ConstantUnitary(
            np.sqrt(1 / 2)
            * np.array(
                [
                    [1, -np.sqrt(3) / 2],
                    [0, 1 / 2],
                    [1, np.sqrt(3) / 2],
                    [0, 1 / 2],
                ]
            )
        )
    )


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
