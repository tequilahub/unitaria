from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from bequem.qubit_map import QubitMap
from bequem.circuit import Circuit

Uuid = str


class Node(ABC):
    """
    Abstract class for all nodes in the computational graph

    The encoded vector or the action of the encoded matrix can be obtained
    either through matrix arithmetic using `simulate` or as a quantum circuit
    using `circuit`.
    """

    # TODO
    # @abstractmethod
    # def __init__(children: list[Node], data: str | None):
    #     raise NotImplementedError

    # @abstractmethod
    # def children(self) -> list[Node]:
    #     return []

    # def uuid() -> Uuid:
    #     raise NotImplementedError

    # def serialize_data(self) -> str | None:
    #     return None

    @abstractmethod
    def qubits_in(self) -> QubitMap:
        raise NotImplementedError

    @abstractmethod
    def qubits_out(self) -> QubitMap:
        raise NotImplementedError

    @abstractmethod
    def normalization(self) -> float:
        raise NotImplementedError

    def is_vector(self) -> bool:
        self.qubits_in().is_all_zeros()

    @abstractmethod
    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def circuit(self) -> Circuit:
        raise NotImplementedError

    def verify(self):
        if self.is_vector():
            computed = self.compute()
            simulated = self.normalization() * self.qubits_out().project(
                self.circuit().simulate())
            np.testing.assert_allclose(computed, simulated)
        else:
            basis_in = self.qubits_in().enumerate_basis()
            basis_out = self.qubits_out().enumerate_basis()
            computed = np.eye(len(basis_out), len(basis_in), dtype=np.complex64)
            simulated = np.zeros((len(basis_out), len(basis_in)), dtype=np.complex64)
            for (i, b) in enumerate(basis_in):
                computed[:, i] = self.compute(computed[:, i])
                simulated[:, i] = self.normalization() * self.qubits_out().project(
                    self.circuit().simulate(b, backend="qulacs"))
            np.testing.assert_allclose(computed, simulated)
