from abc import ABC

import numpy as np

from qubit_map import QubitMap
from circuit import Circuit, simulate

Uuid = str

"""
Abstract class for all nodes in the computational graph

The encoded vector or the action of the encoded matrix can be obtained either through matrix arithmetic using `simulate` or as a quantum circuit using `circuit`.
"""
class Node(ABC):

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
        self.projection_in().is_all_zeros()

    @abstractmethod
    def compute(self, input: np.array | None=None) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def circuit(self) -> Circuit:
        raise NotImplementedError

    def verify(self) -> bool:
        if self.is_vector():
            computed = self.compute()
            simulated = self.normalization() * self.qubits_out().project(simulate(self.circuit()))
            return np.allclose(computed, simulated)
        else:
            basis = self.qubits_in().enumerate_basis()
            for (i, b) in enumerate(basis):
                v = np.zeros(len(basis))
                v[i] = 1
                computed = self.compute(v)
                simulated = self.normalization() * self.qubits_out().project(simulate(self.circuit(), b))
                if not np.allclose(computed, simulated):
                    return False
            return True
                

class WrapperNode(Node):
    # TODO: caching

    @abstractmethod
    def definition(self) -> Node:
        raise NotImplementedError

    def qubits_in(self) -> QubitMap:
        return self.definition().qubits_in()

    def qubits_out(self) -> QubitMap:
        return self.definition().qubits_out()

    def normalization(self) -> float:
        return self.definition().normalization()

    def compute(self, input: np.array | None=None) -> np.array:
        return self.definition().compute(input)

    def circuit(self) -> Circuit:
        return self.definition().circuit()
