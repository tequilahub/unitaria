import numpy as np
import tequila as tq

from bequem.nodes.node import Node
from bequem.nodes.basic_ops import Adjoint
from bequem.circuit import Circuit
from rich.console import Console
from rich.syntax import Syntax

default_verifier = None


def verify(node: Node, reference: np.ndarray | None = None):
    global default_verifier
    if default_verifier is None:
        default_verifier = Verifier()

    default_verifier.verify(node, reference)

class Verifier:

    def __init__(
        self,
        drill: bool = True,
        up_to_phase: bool = False):

        self.drill = drill
        self.up_to_phase = up_to_phase

    def _verify_circuit_qubits(self, node: Node):
        if node.subspace_in.total_qubits == 0:
            # TODO: Tequila does not support circuits without qubits
            assert node.circuit.tq_circuit.n_qubits == 1
            assert node.subspace_out.total_qubits == 0
        else:
            assert node.circuit.tq_circuit.n_qubits == node.subspace_in.total_qubits
            assert node.circuit.tq_circuit.n_qubits == node.subspace_out.total_qubits

    def _compare_batch_compute(self, node: Node, reference: np.ndarray | None = None):
        batch_computed = node.compute(np.eye(node.subspace_in.dimension, dtype=np.complex128))
        computed = np.zeros_like(batch_computed)
        for i in range(node.subspace_in.dimension):
            input = np.zeros(node.subspace_in.dimension, dtype=np.complex128)
            input[i] = 1
            computed[i, :] = node.compute(input)
        np.testing.assert_allclose(batch_computed, computed, atol=1e-8)
        if reference is not None:
            np.testing.assert_allclose(batch_computed, reference)

    def _compare_compute_simulate(self, node: Node):
        basis_in = node.subspace_in.enumerate_basis()

        for (i, b) in enumerate(basis_in):
            input = np.zeros(node.subspace_in.dimension, dtype=np.complex128)
            input[i] = 1
            computed = node.compute(input)
            simulated = node.normalization * node.subspace_out.project(
                    node.circuit.simulate(b, backend="qulacs"))
            if self.up_to_phase:
                simulated = bring_to_same_phase(computed, simulated)
            np.testing.assert_allclose(simulated, computed, atol=1e-8, err_msg=f"On input {i}:")

    def _verify(self, node: Node, reference: np.ndarray | None = None):

        try:
            self._verify_circuit_qubits(node)
            self._compare_compute_simulate(node)
            self._compare_compute_simulate(Adjoint(node))
            self._compare_batch_compute(node, reference)
            self._compare_batch_compute(Adjoint(node))
        except AssertionError as err:
            raise VerificationError(node) from err

    def verify(self, node: Node, reference: np.ndarray | None = None):
        """
        Verify the correctness of this node.

        Checks that the outputs of :py:func:`qubits_in`, :py:func:`qubits_out`,
        :py:func:`normalization`, and :py:func:`circuit` are sensible and encode
        the same matrix.

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
        Convenience function to recursively call :py:func:`verify` on the
        children of this node.

        You should typically use :py:func:`verify` instead.
        """
        for child in node.children():
            self.find_error(child)
            self._verify(child)

def bring_to_same_phase(a: np.ndarray, b: np.ndarray):
    b_f = b.flatten()
    i = np.argmax(np.abs(b_f))
    return a.flatten()[i] / b_f[i] * b


class VerificationError(Exception):

    def __init__(self, node: Node, circuit: Circuit):
        super().__init__()
        self.node = node

    def __str__(self):
        console = Console(width=60)
        circuit = self.node.circuit.draw()
        with console.capture() as capture:
            console.print(self.node.tree())
            console.print(
                Syntax(circuit, "text", background_color="default"))
        output = capture.get()
        return "\n" + output + f"\nnormalization = {self.node.normalization}"
