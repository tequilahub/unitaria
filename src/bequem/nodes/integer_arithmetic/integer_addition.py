import numpy as np

from ..node import Node
from bequem.subspace import Subspace
from bequem.circuit import Circuit
from bequem.circuits.arithmetic import addition_circuit


class IntegerAddition(Node):
    """
    Node implementing the (wrapping) addition of two integers.

    This is a bilinear operation.

    :ivar source_bits:
        The size of the first register. The first summand can be at most ``2 ** source_bits``
    :ivar target_bits:
        The size of the second register. The addition is performed modulo ``2 ** target_bits``.
    """

    def __init__(self, source_bits: int, target_bits: int):
        # TODO: Restriction is because the ancilla free construction needs two source bits.
        #  I know how to fix this but haven't implemented it yet.
        if source_bits < 2 or target_bits < source_bits:
            raise ValueError()
        self.source_bits = source_bits
        self.target_bits = target_bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"source_bits": self.source_bits, "target_bits": self.target_bits}

    def _subspace_in(self) -> Subspace:
        return Subspace(self.source_bits + self.target_bits)

    def _subspace_out(self) -> Subspace:
        return Subspace(self.source_bits + self.target_bits)

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        old_shape = input.shape
        N = 2**self.source_bits
        M = 2**self.target_bits
        input = input.reshape((-1, M, N))
        result = np.zeros_like(input)
        for val in range(N):
            result[:, :, val] += np.roll(input[:, :, val], val, axis=-1)
        return result.reshape(old_shape)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        old_shape = input.shape
        N = 2**self.source_bits
        M = 2**self.target_bits
        input = input.reshape((-1, M, N))
        result = np.zeros_like(input)
        for val in range(N):
            result[:, :, val] += np.roll(input[:, :, val], -val, axis=-1)
        return result.reshape(old_shape)

    def _circuit(self) -> Circuit:
        source = list(reversed(range(self.source_bits)))
        target = list(reversed(range(self.source_bits, self.source_bits + self.target_bits)))
        circuit = addition_circuit(source, target)
        return Circuit(circuit)
