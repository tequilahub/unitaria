import numpy as np

from unitaria.nodes.constants.constant_unitary import ConstantUnitary
from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.node import Node
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.subspace import Subspace


class ConstantMatrix(ProxyNode):
    """
    Node representing the given matrix

    For vectors, `~unitaria.nodes.constant.constant_vector.ConstantVector` is
    more efficient.

    :param matrix: The matrix which should be implemented
    """

    matrix: np.ndarray

    def __init__(self, matrix: np.ndarray):
        super().__init__(matrix.shape[1], matrix.shape[0])
        assert matrix.ndim == 2
        self.matrix = matrix

    def definition(self) -> Node:
        u, s, vh = np.linalg.svd(self.matrix)

        normalization = np.max(np.abs(s))
        s /= normalization

        # extend matrices to be square and power of 2
        n = int(np.ceil(np.log2(np.max(self.matrix.shape))))
        u_ext = np.eye(2**n)
        u_ext[: u.shape[0], : u.shape[1]] = u
        u = u_ext
        vh_ext = np.eye(2**n)
        vh_ext[: vh.shape[0], : vh.shape[1]] = vh
        vh = vh_ext
        s_ext = np.zeros(2**n)
        s_ext[: s.shape[0]] = s
        s = s_ext

        # TODO: There is likely a way to use this structure
        unitary_s = np.block([[np.diag(s), np.diag(np.sqrt(1 - s**2))], [np.diag(-np.sqrt(1 - s**2)), np.diag(s)]])

        U = ConstantUnitary(u)
        S = ConstantUnitary(unitary_s)
        Vh = ConstantUnitary(vh)
        Id2 = Identity(Subspace("#"))
        total_subspace = Subspace("#" * (n + 1))
        subspace_in = Subspace.from_dim(self.matrix.shape[1], bits=total_subspace.total_qubits)
        subspace_out = Subspace.from_dim(self.matrix.shape[0], bits=total_subspace.total_qubits)
        Pin = Projection(total_subspace, subspace_in)
        Pout = Projection(total_subspace, subspace_out)

        return normalization * Pout @ (Id2 & U) @ S @ (Id2 & Vh) @ Pin.adjoint()

    def parameters(self) -> dict:
        return {"matrix": self.matrix}

    def compute(self, input: np.ndarray) -> np.ndarray:
        return (self.matrix @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return (np.conj(self.matrix.T) @ input.T).T


def _extend_basis_by_one(U: np.array, n: int):
    """
    Extends the basis of a (possibly rectangular) unitary matrix by one column/row.

    :param U: The matrix to extend (in-place).
    :param n: The index at which to extend the basis.
    """
    candidates = np.eye(U.shape[0]) - U[:, :n] @ np.conj(U.T)[:n, :]
    norms = np.linalg.norm(candidates, ord=2, axis=0)
    best = np.argmax(norms)
    U[:, n] = candidates[:, best] / norms[best]
