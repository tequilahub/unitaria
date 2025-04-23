from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import tequila as tq
from rich.tree import Tree
from rich.console import Console

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

    # TODO: How do we check that this is implemented properly
    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {}

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
        return self.qubits_in().is_trivial()

    @abstractmethod
    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def circuit(self) -> Circuit:
        raise NotImplementedError

    def tree_label(self, verbose: bool = False):
        label = self.__class__.__name__
        parameters = self.parameters()
        if len(parameters) != 0:
            label += str(parameters)
        return label

    def tree(self, verbose: bool = False, tree: Tree | None = None):
        label = self.tree_label(verbose)
        if tree is None:
            subtree = Tree(label)
        else:
            subtree = tree.add(label)

        for child in self.children():
            child.tree(verbose, subtree)

        return subtree

    def draw(self, verbose: bool = False):
        console = Console()
        with console.capture() as capture:
            console.print(self.tree(verbose))
        output = capture.get()
        return output

    def __str__(self):
        return self.draw()

    def verify_recursive(self, print_circuit: bool = True) -> np.ndarray:
        for child in self.children():
            child.verify_recursive(False)
        return self.verify()

    def verify(self, print_circuit: bool = True) -> np.ndarray:
        basis_in = self.qubits_in().enumerate_basis()
        basis_out = self.qubits_out().enumerate_basis()
        circuit = self.circuit()
        if self.qubits_in().total_qubits == 0:
            # TODO: Tequila does not support circuits without qubits
            assert circuit.tq_circuit.n_qubits == 1
            assert self.qubits_out().total_qubits == 0
        else:
            assert circuit.tq_circuit.n_qubits == self.qubits_in().total_qubits
            assert circuit.tq_circuit.n_qubits == self.qubits_out().total_qubits

        if print_circuit:
            print(self)
            print(tq.draw(circuit.padded(), backend="cirq"))

        if not self.is_vector():
            computed = np.eye(len(basis_out),
                              len(basis_in),
                              dtype=np.complex64)
            simulated = np.zeros((len(basis_out), len(basis_in)),
                                 dtype=np.complex64)
            computed_m = self.compute(computed.T).T
            computed_adj_m = self.compute_adjoint(computed).T
            computed_adj = np.eye(len(basis_in),
                                  len(basis_out),
                                  dtype=np.complex64)

            for (i, b) in enumerate(basis_in):
                computed[:, i] = self.compute(computed[:, i])
                simulated[:, i] = self.normalization() * self.qubits_out(
                ).project(circuit.simulate(b, backend="qulacs"))

            for (i, b) in enumerate(basis_out):
                computed_adj[:, i] = self.compute_adjoint(computed_adj[:, i])

            # verify compute with tensor valued input
            np.testing.assert_allclose(computed, computed_m)
            # verify circuit
            np.testing.assert_allclose(computed, simulated)
            # verify compute_adjoint
            np.testing.assert_allclose(computed_adj_m, np.conj(computed_m).T)
            np.testing.assert_allclose(computed_adj, computed_adj_m)

            return computed_m
        else:
            computed = self.compute(None)
            if computed is None:
                computed = np.array([1])
            simulated = self.normalization() * self.qubits_out().project(
                circuit.simulate(0, backend="qulacs"))

            # verify circuit
            np.testing.assert_allclose(computed, simulated)
            return computed
