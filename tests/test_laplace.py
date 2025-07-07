import pytest

import numpy as np

from bequem.nodes import Increment, Identity, Add, Scale, Adjoint, ConstantUnitary, BlockHorizontal
from bequem.subspace import Subspace
from bequem.verifier import verify


def ref_Q(L, L_MAX):
    """
    Transfer matrix of DP0 spaces from level L to level L_MAX
    """
    N = 2**L - 1
    M = 2**L_MAX - 1

    step = 2 ** (L_MAX - L)

    j = np.arange(1, step)
    stencil = np.zeros(2 * step - 1)
    stencil[step - 1] = 1
    stencil[step - 1 - j] = (step - j) / step
    stencil[step - 1 + j] = (step - j) / step

    j, i = np.meshgrid(np.arange(len(stencil)), np.arange(N))
    indices = (j + i * step).flatten()
    data = np.repeat([stencil], N, axis=0).flatten()

    Q = np.zeros((N, M))
    Q[np.repeat(np.arange(N), 2 * step - 1), indices] = data

    return Q


def ref_P(L, DIM):
    """
    The rectangular preconditioner (F in the paper)
    """
    P = []
    for j in reversed(range(1, L + 1)):
        Q_mat = ref_Q(j, L)

        X = np.array([[1]])
        for d in range(DIM):
            X = np.kron(X, Q_mat)

        P.append(2 ** (-j * (2 - DIM) / 2) * X)
    P = np.vstack(list(reversed(P)))

    return P.T


def ref_S_1d(L):
    """
    1d stiffness matrix
    """
    N = 2**L - 1
    return 2**L * (2 * np.eye(N) - 1 * (np.eye(N, k=1) + np.eye(N, k=-1)))


def ref_C_1d(L):
    """
    1d gradient
    """
    N = 2**L - 1
    return 2 ** (L / 2) * (np.eye(N + 1, N) - np.eye(N + 1, N, k=-1))


def ref_CM_1d(L):
    """
    1d "half" mass matrix
    """
    N = 2**L - 1
    a = 1 / (2 * np.sqrt(3))
    b = 1 / 2
    return 2**L * np.tensordot(np.array([b, a]), np.eye(N + 1, N), 0) + np.tensordot(
        np.array([b, -a]), np.eye(N + 1, N, k=-1), 0
    )


def ref_M_1d(L):
    """
    1d mass matrix
    """
    N = 2**L - 1
    return 2 ** (2 * L) * (2 / 3) * np.eye(N) + (1 / 6) * (np.eye(N, k=1) + np.eye(N, k=-1))


def ref_C(L, DIM):
    """
    D-dimensional gradient
    """
    N = (2**L - 1) ** DIM

    C1d = np.tensordot(np.array([1, 0]), ref_C_1d(L), 0)
    M1d = ref_CM_1d(L)

    Ys = []
    for i in range(DIM):
        Y = np.array([[[1]]])
        for j in range(DIM):
            if i == j:
                Y = np.kron(C1d, Y)
            else:
                Y = np.kron(M1d, Y)
        Ys.append(np.reshape(Y, (-1, N)))
    X = np.vstack(Ys)

    return X


def ref_S(L, DIM):
    """
    D-dimensional stiffenss matrix
    """
    N = (2**L - 1) ** DIM
    X = np.zeros((N, N))

    S1d = ref_S_1d(L)
    M1d = ref_M_1d(L)

    for i in range(DIM):
        Y = np.array([[1]])
        for j in range(DIM):
            if i == j:
                Y = np.kron(Y, S1d)
            else:
                Y = np.kron(Y, M1d)
        X += Y

    return X


def ref_PAP(L, DIM):
    """
    symetrically preconditioned stiffness matrix
    """
    P = ref_P(L, DIM)
    S = ref_S(L, DIM)
    return P.T @ S @ P


def ref_CP(L, DIM):
    """
    Preconditioned gradient
    """
    P = ref_P(L, DIM)
    C = ref_C(L, DIM)
    return C @ P


def ref_CP_1d(L, DIM):
    """
    Preconditioned 1-dimensional gradient

    This has a slightly different indexing than ref_CP(L, 1)
    """
    P = ref_P(L, DIM)
    C = ref_C_1d(L)
    return C @ P


@pytest.mark.parametrize("n", range(1, 5))
def test_laplace(n: int):
    C = Increment(n)
    A = Add(Scale(Identity(Subspace(n)), -2), Scale(Add(C, Adjoint(C)), 1))
    verify(A)


def test_preconditioned_laplace_1d():
    L = 4

    T = None
    C_F = None

    for l in range(L, 0, -1):
        I_l = Identity(Subspace.from_dim(2**l - 1, bits=l), Subspace(l))
        N_l = Increment(l) @ I_l
        C_l = 2 ** (l / 2) * (I_l - N_l)

        T_l = ConstantUnitary(np.sqrt(1 / 2) * np.array([[1], [1]])) & Identity(Subspace(l - 1))
        # T_l = ConstantUnitary(
        #     np.sqrt(1 / 2) * np.array([
        #         [1, -np.sqrt(3) / 2],
        #         [0, 1 / 2],
        #         [1, np.sqrt(3) / 2],
        #         [0, 1 / 2],
        #     ])) & Identity(Subspace(l - 1))

        if l == L:
            TC = 2 ** (-l / 2) * C_l
            C_F = TC
            T = T_l
        else:
            TC = 2 ** (-l / 2) * T @ C_l
            C_F = BlockHorizontal(TC, C_F)
            T = T @ T_l

    verify(C_F, ref_CP_1d(4, 1).T)
