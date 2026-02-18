"""
Module for verifying the correctess of nodes.
"""

from typing import Sequence

import numpy as np

from unitaria.nodes.node import Node
from unitaria.nodes.basic.adjoint import Adjoint
from rich.console import Console
from rich.syntax import Syntax

default_verifier = None


def verify(node: Node, reference: np.ndarray | None = None):
    """
    Verify a node using the default `Verifier`.

    This checks for ``node`` that

    * The dimesions of `~unitaria.nodes.Node.subspace_in` and
      `~unitaria.nodes.Node.subspace_out` match `~unitaria.nodes.Node.dimension_in`
      and `~unitaria.nodes.Node.dimension_out`, and the total qubits of these
      subspaces matches the qubits in `~unitaria.nodes.Node.circuit`.
    * `~unitaria.nodes.Node.compute` and `~unitaria.nodes.Node.compute_adjoint`
      match and that their batched version are equivalent.
    * `~unitaria.nodes.Node.compute` matches the circuit
      implementation given through `~unitaria.nodes.Node.subspace_in`,
      `~unitaria.nodes.Node.subspace_out`, and `~unitaria.nodes.Node.circuit`.

    If one of these checks does not pass, the same tests will be run for all
    children to find out whether the implementation of this node, or one of its
    childrens is erroneous.

    :param node:
        The node to be checked.
    :param reference:
        An optional reference, to which the circuit and matrix arithmetic
        implementations should be compared. For example for ``Identity(1)`` one
        could pass ``np.eye(2)``.
    """
    global default_verifier
    if default_verifier is None:
        default_verifier = Verifier()

    nodes = [node]
    if isinstance(node, Sequence):
        nodes = node
    for node in nodes:
        default_verifier.verify(node, reference)


class Verifier:
    """
    Object for verifying nodes.

    See `verify` for the checks that are performed.

    :param drill:
        If this is ``True``, upon encountering an error in a node, the
        children of this node are checked recursively, to find out whether the
        implementation of this node, or one of its childrens is erroneous.
    :param up_to_phase:
        If this is ``True``, the output from circuit simulation is checked only
        up to a multiplicative global phase ``np.exp(theta * 1j)``. This might
        be useful, since some simulation backends to not guarantee to simulate
        the global phase correctly.
    """

    drill: bool
    up_to_phase: bool

    def __init__(self, drill: bool = True, up_to_phase: bool = False):
        self.drill = drill
        self.up_to_phase = up_to_phase

    def _verify_circuit_subspaces(self, node: Node):
        assert node.dimension_in == node.subspace_in.dimension
        assert node.dimension_out == node.subspace_out.dimension
        expected_qubits = node.target_qubit_count() + node.clean_ancilla_count() + node.borrowed_ancilla_count()
        if expected_qubits == 0:
            # TODO: Tequila does not support circuits without qubits
            expected_qubits = 1
        assert node.circuit().n_qubits == expected_qubits

    def _compare_batch_compute(self, node: Node, reference: np.ndarray | None = None):
        batch_computed = node.toarray(force_matrix=True)
        computed = np.zeros_like(batch_computed.T)
        for i in range(node.dimension_in):
            input = np.zeros(node.dimension_in, dtype=np.complex128)
            input[i] = 1
            computed[i, :] = node.compute(input)
        np.testing.assert_allclose(batch_computed, computed.T, atol=1e-8)
        if reference is not None:
            if reference.ndim == 1:
                reference = np.expand_dims(reference, 1)
            np.testing.assert_allclose(batch_computed, reference)

    def _compare_compute_simulate(self, node: Node):
        basis_in = node.subspace_in.enumerate_basis()

        # TODO: Could now use Node.simulate, but this wont work with `up_to_phase`
        for i, b in enumerate(basis_in):
            input = np.zeros(node.subspace_in.dimension, dtype=np.complex128)
            input[i] = 1
            computed = node.compute(input)
            simulated = node.normalization * node.subspace_out.project(node.circuit().simulate(b, backend="qulacs"))
            if self.up_to_phase:
                simulated = _bring_to_same_phase(computed, simulated)
            np.testing.assert_allclose(simulated, computed, atol=1e-8, err_msg=f"On input {i}:")

    def _verify(self, node: Node, reference: np.ndarray | None = None):
        try:
            self._verify_circuit_subspaces(node)
            self._compare_compute_simulate(node)
            self._compare_compute_simulate(Adjoint(node))
            self._compare_batch_compute(node, reference)
            self._compare_batch_compute(Adjoint(node))
        except AssertionError as err:
            raise VerificationError(node) from err

    def verify(self, node: Node, reference: np.ndarray | None = None):
        """
        Verify the correctness of this node.

        Checks that the outputs of `qubits_in`, `qubits_out`, `normalization`,
        and `circuit` are sensible and encode the same matrix.

        :param drill:
            If True and an error is found, recursivly test this nodes children
            to find the smallest node which still contains the error.
        """

        try:
            self._verify(node, reference)
        except VerificationError as err:
            if self.drill:
                self.find_error(node)
            raise err

    def find_error(self, node: Node):
        """
        Convenience function to recursively call `verify` on the
        children of this node.

        You should typically use `verify` instead.
        """
        for child in node.children():
            self.find_error(child)
            self._verify(child)


def _bring_to_same_phase(a: np.ndarray, b: np.ndarray):
    b_f = b.flatten()
    i = np.argmax(np.abs(b_f))
    return a.flatten()[i] / b_f[i] * b


class VerificationError(Exception):
    """
    Exception thrown during node verification.

    :param node:
        The node which was found to be invalid.
    """

    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        try:
            self._circuit = self.node.circuit().draw()
            self._tree = self.node.tree()
            self._normalization = self.node.normalization
        except Exception:
            self._circuit = None

    def __str__(self):
        if self._circuit is None:
            return "(Could not render circuit)"
        console = Console(width=60)
        with console.capture() as capture:
            console.print(self._tree)
            console.print(Syntax(self._circuit, "text", background_color="default"))
        output = capture.get()
        return "\n" + output + f"\nnormalization = {self._normalization}"
