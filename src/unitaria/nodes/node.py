from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from rich.tree import Tree
from rich.console import Console

from unitaria.subspace import Subspace
from unitaria.circuit import Circuit


class Node(ABC):
    """
    Abstract class for all nodes in the computational graph

    The encoded vector or the action of the encoded matrix can be obtained
    either through matrix arithmetic using `compute` or as a quantum
    block encoding using `circuit`, `normalization`, `subspace_in`, and
    `subspace_out`.

    When creating your own subclass of `Node`, you should implement the
    functions `_subspace_in`, `_subspace_out`, `_normalization`, `_circuit`,
    `compute`, and `compute_adjoint`. Potentially also `children` and
    `parameters`.

    Note the prefixed underscores for some of these methods, which are necessary
    due to how the properties of `Node` are cached.

    To check that these methods are implemented correctly, use
    `~unitaria.verifier.verify`.

    As an alternative if you can represent your custom node in terms of other
    nodes, and just want to provide a more efficient implementation of some
    of the functions -- say `compute` and `compute_adjoint` -- you can instead
    create a subclass of `~unitaria.nodes.ProxyNode`.
    """

    def __init__(self, dimension_in: int, dimension_out: int):
        self.dimension_in = dimension_in
        self.dimension_out = dimension_out

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
        space nodes circuit, which has dimension :math:`2^n` where :math:`n` is
        the number of qubits. In other words, this defines whether a particular
        basis state is "valid" or "invalid".

        In the formalism of block encodings this corresponds to the projection
        :math:`\\Pi_1`.
        """
        return self._subspace_in()

    @abstractmethod
    def _subspace_in(self) -> Subspace:
        """
        Method for computing `subspace_in`.

        To be implemented in all subclasses of `Node`.
        """
        raise NotImplementedError

    @cached_property
    def subspace_out(self) -> Subspace:
        """
        The embedding of the output vectorspace.

        Specifically this specifies how the vectorspace is included in state
        space nodes circuit, which has dimension :math:`2^n` where :math:`n` is
        the number of qubits. In other words, this defines whether a particular
        basis state is "valid" or "invalid".

        In the formalism of block encodings this corresponds to the projection
        :math:`\\Pi_2`.
        """
        return self._subspace_out()

    @abstractmethod
    def _subspace_out(self) -> Subspace:
        """
        Method for computing `subspace_out`.

        To be implemented in all subclasses of `Node`.
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
        Method for computing `normalization`.

        To be implemented in all subclasses of `Node`.
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

        Input may be a vector or a higher order tensor. If it is a vector
        it will have dimension equal to the dimension of `qubits_in`. If
        it is a tensor, the last dimension will be equal to the dimension
        of `qubits_in`. In this case, the operation should be applied to
        all vectors ``input[i, j, ..., k, :]`` in parallel. The shape of
        the returned array should match the input shape in all but the last
        dimension.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        """
        Apply the adjoint action of this nodes matrix to the input.

        See `compute` for input and output formats.
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
        Method for computing `circuit`.

        To be implemented in all subclasses of `Node`.
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

    def toarray(self, force_matrix: bool = False) -> np.ndarray:
        """
        Returns a numpy array representing this node, as given by `compute`.

        The output of this method should match `simulate`, which can be checked
        using `~unitaria.verifier.verify`.

        By default this will return a one-dimension array if the node represents
        a vector. However, this behaviour can be suppressed by setting
        ``force_matrix = True``

        :param force_matrix:
            Force the method to return a two-dimensional array, even when the
            node corresponds to a vector.
        """
        if self.is_vector() and not force_matrix:
            return self.compute(np.array([1], dtype=np.complex128))
        else:
            return self.compute(np.eye(self.subspace_in.dimension, dtype=np.complex128)).T

    def compute_norm(self, input: np.array | None) -> float:
        """
        Method to compute the norm of the wavefunction in the subspace.
        """
        if input is None and not self.is_vector():
            raise ValueError("Cannot compute norm, since node is a matrix and no input was given")
        if self.is_vector() and input is None:
            input = np.array([1])
        return np.linalg.norm(self.compute(input))

    def simulate(self, input: int | None = None) -> np.ndarray:
        """
        Returns a numpy array representing this node, as given by `simulate`.

        When ``input`` is ``None``, the output of this method should match
        `compute`, which can be checked using `~unitaria.verifier.verify`.

        :param input:
            The index of the optional initial state, with which this node
            should be simulated. Should be a number between ``0`` and
            ``node.subspace_out.dimension - 1``
        """
        if self.is_vector() and input is None:
            input = 0
        if input is not None:
            wavefunction = self.circuit.simulate(input=input)
            return self.normalization * self.subspace_out.project(wavefunction)
        else:
            output = np.zeros((self.subspace_out.dimension, self.subspace_in.dimension))
            for i, b in enumerate(self.subspace_in.enumerate_basis()):
                input = np.zeros(self.subspace_in.dimension, dtype=np.complex128)
                input[i] = 1
                output[i] = self.normalization * self.subspace_out.project(self.circuit.simulate(b, backend="qulacs"))
            return output

    def simulate_norm(self, input: int | None = None) -> float:
        """
        Method to simulate the norm of the encoded vector in the subspace using a circuit.

        When calling this function on a node encoding a matrix, ``input`` has to
        be a vector. In this case the norm of the matrix applied to ``input`` is given.

        :param input:
            A vector to which the encoded matrix is applied before computing the norm.
        """
        if input is None and not self.is_vector():
            raise ValueError("Cannot simulate norm, since node is a matrix and no input was given")
        norm = np.linalg.norm(self.simulate(input))
        return norm

    def tree(self, verbose: bool = False, tree: Tree | None = None, holes: list[Node] = []) -> Tree:
        """
        Method for rich text output of the computational graph.

        Typically you should call `draw` instead of this method.
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
            `~unitaria.nodes.proxy_node.ProxyNode` is inserted into the
        """
        console = Console()
        with console.capture() as capture:
            console.print(self.tree(verbose), end="")
        output = capture.get().strip()
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
