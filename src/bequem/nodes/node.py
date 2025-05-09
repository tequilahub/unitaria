from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import tequila as tq
from rich.tree import Tree
from rich.console import Console
from rich.syntax import Syntax

from bequem.qubit_map import Subspace
from bequem.circuit import Circuit

Uuid = str


class Node(ABC):
    """
    Abstract class for all nodes in the computational graph

    The encoded vector or the action of the encoded matrix can be obtained
    either through matrix arithmetic using :py:func:`compute` or as a
    quantum circuit using :py:func:`circuit`.
    """

    # TODO
    # @abstractmethod
    # def __init__(children: list[Node], data: str | None):
    #     raise NotImplementedError

    # TODO: How do we check that this is implemented properly
    def children(self) -> list[Node]:
        """
        The children nodes of this node

        Mostly used for serialization and systematic verification of the
        computational graph. Children and parameters together should fully
        define the matrix encoded by any node.
        """
        return []

    def parameters(self) -> dict:
        """
        The parameters of this graph

        Mostly used for serialization and systematic verification of the
        computational graph. Children and parameters together should fully
        define the matrix encoded by any node.
        """
        return {}

    # def uuid() -> Uuid:
    #     raise NotImplementedError

    # def serialize_data(self) -> str | None:
    #     return None

    @abstractmethod
    def subspace_in(self) -> Subspace:
        """
        The embedding of the input vectorspace.

        Specifically this specifies how the vectorspace is included in state
        space nodes circuit, which has dimension 2^n where n is the number of qubits.
        In other words, this defines whether a particular basis state is "valid" or "invalid".

        In the formalism of block encodings this corresponds to the projection Pi_1.
        """
        raise NotImplementedError

    @abstractmethod
    def subspace_out(self) -> Subspace:
        """
        The embedding of the output vectorspace.

        Specifically this specifies how the vectorspace is included in state
        space nodes circuit, which has dimension 2^n where n is the number of qubits.
        In other words, this defines whether a particular basis state is "valid" or "invalid".

        In the formalism of block encodings this corresponds to the projection Pi_2.
        """
        raise NotImplementedError

    @abstractmethod
    def normalization(self) -> float:
        """
        Normalization of the block encoding.

        Non-negative number, which has to be multiplied with the outputs of the
        circuit to ensure proper scaling of the result.
        """
        raise NotImplementedError

    @abstractmethod
    def phase(self) -> float:
        """
        The global phase of the block encoding which can't be represented in the Tequila circuit.

        If a global phase gate is added to Tequila in the future, that should be used instead.

        The output of the circuit has to be multiplied with exp(i * phase) to get the correct result.
        """
        raise NotImplementedError

    def is_vector(self) -> bool:
        """
        Tests whether this node encodes a vector or a matrix.
        """
        return self.subspace_in().is_trivial()

    @abstractmethod
    def compute(self, input: np.ndarray | None = None) -> np.ndarray:
        """
        Apply the action of this nodes matrix to the input.

        If this node encodes a vector, then ``input = None`` is valid, in which
        case the method should simply return the encoded vector.

        Input may be a vector or a higher order tensor. If it is a vector it
        will have dimension equal to the dimension of :py:func:`qubits_in`.
        If it is a tensor, the last dimension will be equal to the dimension
        of :py:func:`qubits_in`. In this case, the operation should be applied
        to all vectors ``input[i, j, ..., k, :]`` in parallel. The shape of
        the returned array should match the input shape in all but the last
        dimension.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_adjoint(self, input: np.ndarray | None = None) -> np.ndarray:
        """
        Apply the adjoint action of this nodes matrix to the input.

        See :py:func:`compute` for input and output formats.
        """
        raise NotImplementedError

    @abstractmethod
    def circuit(self) -> Circuit:
        """
        The circuit corresponding to the unitary of the block encoding.
        """
        raise NotImplementedError

    def tree_label(self, verbose: bool = False):
        """
        Label by which this node should be represented in textual (debug)
        output.

        Defaults to the name of the nodes class plus its parameters.
        """
        label = self.__class__.__name__
        parameters = self.parameters()
        if len(parameters) != 0:
            label += str(parameters)
        return label

    def tree(self,
             verbose: bool = False,
             tree: Tree | None = None,
             holes: list[Node] = []) -> Tree:
        """
        Method for rich text output of the computational graph.

        Typically you should call :py:func:`draw` instead of this method.
        """
        for (i, hole) in enumerate(holes):
            if hole is self:
                return tree.add(f"child {i}")

        label = self.tree_label(verbose)
        if tree is None:
            subtree = Tree(label)
        else:
            subtree = tree.add(label)

        for child in self.children():
            child.tree(verbose, subtree, holes)

        return subtree

    def draw(self, verbose: bool = False) -> str:
        """
        Rich text output of the computational graph

        :param verbose:
            if set to ``True``, the definition of any
            :py:class:`~bequem.nodes.proxy_node.ProxyNode` is inserted into
            the output.
        """
        console = Console()
        with console.capture() as capture:
            console.print(self.tree(verbose))
        output = capture.get()
        return output

    def __str__(self):
        return self.draw()

    def find_error(self) -> np.ndarray:
        """
        Convenience function to recursively call :py:func:`verify` on the
        children of this node.

        You should typically use :py:func:`verify` instead.
        """
        for child in self.children():
            child.find_error()

    def verify(self, drill: bool = True) -> np.ndarray:
        """
        Verify the correctness of this node.

        Checks that the outputs of :py:func:`qubits_in`, :py:func:`qubits_out`,
        :py:func:`normalization`, and :py:func:`circuit` are sensible and encode
        the same matrix.

        :param drill:
            If True and an error is found, recursivly test this nodes children
            to find the smallest node which still contains the error.
        """
        basis_in = self.subspace_in().enumerate_basis()
        basis_out = self.subspace_out().enumerate_basis()
        circuit = self.circuit()
        try:
            if self.subspace_in().total_qubits == 0:
                # TODO: Tequila does not support circuits without qubits
                assert circuit.tq_circuit.n_qubits == 1
                assert self.subspace_out().total_qubits == 0
            else:
                assert circuit.tq_circuit.n_qubits == self.subspace_in(
                ).total_qubits
                assert circuit.tq_circuit.n_qubits == self.subspace_out(
                ).total_qubits

            if not self.is_vector():
                computed = np.eye(len(basis_out),
                                  len(basis_in),
                                  dtype=np.complex64)
                simulated = np.zeros((len(basis_out), len(basis_in)),
                                     dtype=np.complex64)
                computed_m = self.compute(np.eye(len(basis_in))).T
                computed_adj_m = self.compute_adjoint(np.eye(len(basis_out))).T
                computed_adj = np.eye(len(basis_in),
                                      len(basis_out),
                                      dtype=np.complex64)

                for (i, b) in enumerate(basis_in):
                    input = np.zeros(len(basis_in))
                    input[i] = 1
                    computed[:, i] = self.compute(input)
                    simulated[:, i] = np.exp(
                        self.phase() *
                        1j) * self.normalization() * self.subspace_out().project(
                            circuit.simulate(b, backend="qulacs"))

                for (i, b) in enumerate(basis_out):
                    input = np.zeros(len(basis_out))
                    input[i] = 1
                    computed_adj[:, i] = self.compute_adjoint(input)

                # verify compute with tensor valued input
                np.testing.assert_allclose(computed, computed_m, atol=1e-8)
                # verify circuit
                np.testing.assert_allclose(computed, simulated, atol=1e-8)
                # verify compute_adjoint
                np.testing.assert_allclose(computed_adj_m,
                                           np.conj(computed_m).T, atol=1e-8)
                np.testing.assert_allclose(computed_adj, computed_adj_m, atol=1e-8)

                return computed_m
            else:
                computed = self.compute(None)
                if computed is None:
                    computed = np.array([1])
                simulated = np.exp(
                    self.phase() *
                    1j) * self.normalization() * self.subspace_out().project(
                        circuit.simulate(0, backend="qulacs"))

                # verify circuit
                np.testing.assert_allclose(computed, simulated, atol=1e-8)
                return computed
        except AssertionError as err:
            if drill:
                try:
                    self.find_error()
                except VerificationError as child_err:
                    raise VerificationError(self, circuit) from child_err
            raise VerificationError(self, circuit) from err


class VerificationError(Exception):

    def __init__(self, node: Node, circuit: Circuit):
        super().__init__()
        self.node = node
        compiled = tq.simulators.simulator_api.compile_circuit(
            abstract_circuit=circuit.padded(), backend="cirq")
        self.circuit = compiled.circuit.to_text_diagram()

    def __str__(self):
        console = Console(width=60)
        with console.capture() as capture:
            console.print(self.node.tree())
            console.print(
                Syntax(self.circuit, "text", background_color="default"))
        output = capture.get()
        return "\n" + output + f"\nnormalization = {self.node.normalization()}, phase = {self.node.phase()}"
