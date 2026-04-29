import numpy as np
import scipy.sparse as sp
import unitaria as ut


def mat_Q(coarse_size, L, L_MAX):
    """
    The matrix reperesentation of V_L -> V_{L_MAX}
    """
    N = coarse_size * 2**L - 1
    M = coarse_size * 2**L_MAX - 1

    step = 2 ** (L_MAX - L)

    j = np.arange(1, step)
    stencil = np.zeros(2 * step - 1)
    stencil[step - 1] = 1
    stencil[step - 1 - j] = (step - j) / step
    stencil[step - 1 + j] = (step - j) / step

    j, i = np.meshgrid(np.arange(len(stencil)), np.arange(N))
    indices = (j + i * step).flatten()
    indptr = np.arange(N + 1) * (2 * step - 1)
    data = np.repeat([stencil], N, axis=0).flatten()

    return sp.csr_array((data, indices, indptr), shape=(N, M)).T


def mat_P(coarse_size, L, DIM):
    """
    The rectangular preconditioner (F in the paper)
    """
    P = []
    for j in reversed(range(1, L + 1)):
        Q_mat = mat_Q(coarse_size, j, L)

        X = sp.csr_array([[1]])
        for d in range(DIM):
            X = sp.kron(X, Q_mat)

        P.append(2 ** (-j * (2 - DIM) / 2) * X)
    P = sp.hstack(P)

    return P


# In the following functions, L is just used as the scaling factor. This allows
# to assemble matrices with and without boundary elements.


def mat_C_1d(N_cells, fine_level, incl_bd=False):
    """
    1d gradient
    """
    offsets = [1, 0] if incl_bd else [0, -1]
    Nin = N_cells + 1 if incl_bd else N_cells - 1
    X = 2 ** (fine_level / 2) * sp.diags_array([1.0, -1], offsets=offsets, shape=(N_cells, Nin))
    Y = 0 * np.eye(N_cells, Nin)
    return sp.vstack((X, Y))


def mat_C_1d_rescaled(h, N_cells, fine_level, incl_bd=False):
    """
    1d gradient
    """
    offsets = [1, 0] if incl_bd else [0, -1]
    Nin = N_cells + 1 if incl_bd else N_cells - 1
    X = 1 / h * sp.diags_array([1.0, -1], offsets=offsets, shape=(N_cells, Nin))
    Y = 0 * np.eye(N_cells, Nin)
    return sp.vstack((X, Y))


def mat_CM_1d(N_cells, fine_level, incl_bd=False):
    """
    1d "half" mass matrix
    """
    offsets = [1, 0] if incl_bd else [0, -1]
    Nin = N_cells + 1 if incl_bd else N_cells - 1
    a = 1 / (2 * np.sqrt(3))
    b = 1 / 2
    X = b * sp.diags_array([1.0, 1], offsets=offsets, shape=(N_cells, Nin))
    Y = a * sp.diags_array([1.0, -1], offsets=offsets, shape=(N_cells, Nin))
    return 2 ** (-fine_level / 2) * sp.vstack((X, Y))


def mat_CM_1d_rescaled(N_cells, fine_level, incl_bd=False):
    """
    1d "half" mass matrix
    """
    offsets = [1, 0] if incl_bd else [0, -1]
    Nin = N_cells + 1 if incl_bd else N_cells - 1
    a = 1 / (2 * np.sqrt(3))
    b = 1 / 2
    X = b * sp.diags_array([1.0, 1], offsets=offsets, shape=(N_cells, Nin))
    Y = a * sp.diags_array([1.0, -1], offsets=offsets, shape=(N_cells, Nin))
    return sp.vstack((X, Y))


def mat_C(N_cells, fine_level, DIM, incl_bd=False):
    """
    D-dimensional gradient
    """

    C1d = mat_C_1d(N_cells, fine_level, incl_bd)
    M1d = mat_CM_1d(N_cells, fine_level, incl_bd)

    Ys = []
    for i in range(DIM):
        Y = sp.eye(1, 1)
        for j in range(DIM):
            if i == j:
                Y = sp.kron(C1d, Y)
            else:
                Y = sp.kron(M1d, Y)
        Ys.append(Y)
    X = sp.vstack(Ys)

    return X


def mat_CP(coarse_size, L, DIM):
    """
    Preconditioned gradient
    """
    P = mat_P(coarse_size, L, DIM)
    C = mat_C(coarse_size * 2**L, L, DIM)
    return C @ P


def preconditioned_half_laplace(
    coarse_size,
    fine_level,
    dim,
):
    T = None
    C_F = None

    H = ut.ConstantUnitary(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]))
    M = (1 / np.sqrt(2)) * ut.ComponentwiseMul(ut.ConstantVector(np.array([1, 1 / np.sqrt(3)]))) @ H

    coarse_bits = int(np.ceil(np.log2(coarse_size)))

    for level in range(fine_level, 0, -1):
        dimension_in = coarse_size * 2**level - 1
        dimension_out = coarse_size * 2**level
        subspace_out = ut.Subspace.from_dim(dimension_out, bits=level + coarse_bits)
        I_l = ut.Identity(subspace_out)[:, :dimension_in]
        N_l = ut.Increment(bits=level + coarse_bits)[:dimension_out, :dimension_in]

        R_l_1d = 2 ** (-level / 2) * (M & ut.Identity(subspace_out)) @ ut.BlockVertical(I_l, N_l)
        C_l_1d = 2 ** (level / 2) * ut.ConstantVector(np.array([1, 0])) & (I_l - N_l)

        C_l = None

        for i in range(dim):
            partial = None

            for j in range(dim):
                if j == i:
                    X = C_l_1d
                else:
                    X = R_l_1d
                if partial is None:
                    partial = X
                else:
                    partial = X & partial
            if C_l is None:
                C_l = partial
            else:
                C_l = ut.BlockVertical(C_l, partial)

        T_l_1d = ut.ConstantUnitary(
            np.sqrt(1 / 2)
            * np.array(
                [
                    [1, -np.sqrt(3) / 2],
                    [0, 1 / 2],
                    [1, np.sqrt(3) / 2],
                    [0, 1 / 2],
                ]
            )
        ) & ut.Identity(ut.Subspace.from_dim(coarse_size * 2 ** (level - 1), bits=level - 1 + coarse_bits))

        subspace_out_1 = ut.Subspace("#") & subspace_out
        permute = ut.PermuteFactors(
            subspace_out_1, [len(subspace_out.tensor_factors)] + list(range(len(subspace_out.tensor_factors)))
        )
        T_l_1d = permute @ T_l_1d

        assert T_l_1d.dimension_out == R_l_1d.dimension_out
        assert T_l_1d.dimension_out == C_l_1d.dimension_out

        T_l = T_l_1d
        for i in range(1, dim):
            T_l = T_l & T_l_1d

        if level == fine_level:
            TC = 2 ** ((2 - dim) * (-level / 2)) * C_l
            C_F = TC
            T = T_l
        else:
            T_tilde = ut.Identity(ut.Subspace.from_dim(dim)) & T
            TC = 2 ** ((2 - dim) * (-level / 2)) * T_tilde @ C_l
            C_F = ut.BlockHorizontal(C_F, TC)
            T = T @ T_l

    return C_F


def test_preconditioned_half_laplace():
    ut.verify(preconditioned_half_laplace(1, 2, 2), mat_CP(1, 2, 2).toarray(), check_adjoint=False)
    np.testing.assert_allclose(preconditioned_half_laplace(1, 4, 2).toarray(), mat_CP(1, 4, 2).toarray())


def coefficient_diag(f, fine_level, dim):
    x = np.repeat((np.arange(2**fine_level) + 0.5) * h, 2)
    y = np.repeat((np.arange(2**fine_level) + 0.5) * h, 2)
    coordinates = [x, y]
    values = f(np.meshgrid(*coordinates)).flatten()

    N = (2**fine_level) ** dim * 2**dim

    def compute(x):
        return np.einsum("...j,j->...j", x, values)

    def compute_adjoint(x):
        return np.einsum("...j,j->...j", x, values)

    return ut.AbstractNode(N, N, compute, compute_adjoint, np.max(values)) & ut.Identity(ut.Subspace.from_dim(dim))


def sin_coefficient(X):
    eps = 1 / 8
    return (2 + np.sin(2 * np.pi * X[0] / eps - np.pi / 2)) * (2 + np.sin(2 * np.pi * X[1] / eps - np.pi / 2))


if __name__ == "__main__":
    fine_level = 6
    h = 1 / 2**fine_level

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7, 10), width_ratios=[5, 1])

    # Plot coefficient
    x = (np.arange(2**fine_level) + 0.5) * h
    y = (np.arange(2**fine_level) + 0.5) * h
    coordinates = [x, y]
    values = sin_coefficient(np.meshgrid(*coordinates))
    im0 = axes[0, 0].imshow(values)
    axes[0, 0].set_title("coefficient")
    fig.colorbar(im0, cax=axes[0, 1], shrink=0.8)

    def sqrt_coefficient(x):
        return np.sqrt(sin_coefficient(x))

    # Note that coefficients_diag would not be an efficient implementation.
    # Instead, the coefficient values should be computed on the quantum computer.
    D_half = coefficient_diag(sqrt_coefficient, fine_level, 2)

    CP = preconditioned_half_laplace(1, fine_level, 2)
    S_half = D_half @ CP

    P = mat_P(1, fine_level, 2).toarray()
    Q, _R = np.linalg.qr(P.T)

    # S_half_mat = S_half.toarray()
    # condition = S_half.normalization / np.linalg.norm(S_half_mat @ Q, ord=-2)
    # print(f"condition = {condition}")
    condition = 15

    S_half_inv = ut.Pseudoinverse(S_half, condition, 0.1)
    S_inv = S_half_inv @ S_half_inv.adjoint()

    # This is also cheating
    rhs = np.ones((2**fine_level - 1) ** 2) * h**2
    preconditioned_rhs = P.T @ rhs

    preconditioned_sol = (S_inv @ ut.ConstantVector(preconditioned_rhs)).toarray().real
    sol = P @ preconditioned_sol

    im1 = axes[1, 0].imshow(sol.reshape((2**fine_level - 1, 2**fine_level - 1)))
    axes[1, 0].set_title("solution")
    fig.colorbar(im1, cax=axes[1, 1], shrink=0.8)
    plt.show()
