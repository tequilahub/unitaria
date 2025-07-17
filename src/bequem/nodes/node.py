from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from rich.tree import Tree
from rich.console import Console

from bequem.subspace import Subspace
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

    @cached_property
    def subspace_in(self) -> Subspace:
        """
        The embedding of the input vectorspace.

        Specifically this specifies how the vectorspace is included in state
        space nodes circuit, which has dimension 2^n where n is the number of qubits.
        In other words, this defines whether a particular basis state is "valid" or "invalid".

        In the formalism of block encodings this corresponds to the projection Pi_1.
        """
        return self._subspace_in()

    @abstractmethod
    def _subspace_in(self) -> Subspace:
        """
        Method to for computing :py:func:`subspace_in`.

        To be implemented in all subclasses of :py:class:`Node`.
        """
        raise NotImplementedError

    @cached_property
    def subspace_out(self) -> Subspace:
        """
        The embedding of the output vectorspace.

        Specifically this specifies how the vectorspace is included in state
        space nodes circuit, which has dimension 2^n where n is the number of qubits.
        In other words, this defines whether a particular basis state is "valid" or "invalid".

        In the formalism of block encodings this corresponds to the projection Pi_2.
        """
        return self._subspace_out()

    @abstractmethod
    def _subspace_out(self) -> Subspace:
        """
        Method to for computing :py:func:`subspace_out`.

        To be implemented in all subclasses of :py:class:`Node`.
        """
        raise NotImplementedError

    @cached_property
    def normalization(self) -> float:
        """
        Normalization of the block encoding.

        Non-negative number, which has to be multiplied with the outputs of the
        circuit to ensure proper scaling of the result.
        """
        return self._normalization()

    @abstractmethod
    def _normalization(self) -> float:
        """
        Method to for computing :py:func:`normalization`.

        To be implemented in all subclasses of :py:class:`Node`.
        """
        raise NotImplementedError

    def is_vector(self) -> bool:
        """
        Tests whether this node encodes a vector or a matrix.
        """
        return self.subspace_in.is_trivial()

    @abstractmethod
    def compute(self, input: np.ndarray) -> np.ndarray:
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
    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        """
        Apply the adjoint action of this nodes matrix to the input.

        See :py:func:`compute` for input and output formats.
        """
        raise NotImplementedError

    @cached_property
    def circuit(self) -> Circuit:
        """
        The circuit corresponding to the unitary of the block encoding.
        """
        return self._circuit()

    @abstractmethod
    def _circuit(self) -> float:
        """
        Method to for computing :py:func:`circuit`.

        To be implemented in all subclasses of :py:class:`Node`.
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

    def compute_norm(self, input: np.array) -> float:
        """
        Method to compute the norm of the wavefunction in the subspace.
        """
        result = self.compute(input=input)
        return np.linalg.norm(result)

    def simulate_norm(self, input: np.ndarray | int = 0) -> float:
        """
        Method to simulate the norm of the wavefunction in the subspace using a circuit.
        """
        wavefunction = self.circuit.simulate(input=input)
        norm = np.linalg.norm([wavefunction[i] for i in self.subspace_out.enumerate_basis()])
        return norm

    def tree(self, verbose: bool = False, tree: Tree | None = None, holes: list[Node] = []) -> Tree:
        """
        Method for rich text output of the computational graph.

        Typically you should call :py:func:`draw` instead of this method.
        """
        for i, hole in enumerate(holes):
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

    def __repr__(self):
        out = self.__class__.__name__ + "("
        parameters = self.parameters()
        if len(parameters) != 0:
            out += str(parameters)[1:-1]
            if len(self.children()) > 0:
                out += ", "
        if len(self.children()) > 0:
            out += self.children().__repr__()
        out += ")"
        return out
