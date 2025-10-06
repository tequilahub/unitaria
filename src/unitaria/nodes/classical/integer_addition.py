from typing import Sequence

import numpy as np

from ..node import Node
from .classical import Classical
from unitaria.circuit import Circuit
from unitaria.circuits.arithmetic import addition_circuit


class IntegerAddition(Classical):
    """
    Node implementing the (wrapping) addition of two integers.

    This is a bilinear operation.

    :param source_bits:
        The size of the first register. The first summand can be at most ``2 ** source_bits``
    :param target_bits:
        The size of the second register. The addition is performed modulo ``2 ** target_bits``.
    """

    source_bits: int
    target_bits: int

    def __init__(self, source_bits: int, target_bits: int):
        super().__init__(source_bits + target_bits, source_bits + target_bits)
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

    def compute_classical(self, input: np.ndarray) -> np.ndarray:
        input1 = input % 2**self.source_bits
        input2 = input // 2**self.source_bits
        input2 = (input2 + input1) % 2**self.target_bits
        return input1 + input2 * 2**self.source_bits

    def compute_reverse_classical(self, input: np.ndarray) -> np.ndarray:
        input1 = input % 2**self.source_bits
        input2 = input // 2**self.source_bits
        input2 = (input2 - input1) % 2**self.target_bits
        return input1 + input2 * 2**self.source_bits

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        source = target[: self.source_bits]
        target = target[self.source_bits :]
        circuit = addition_circuit(source, target)
        return Circuit(circuit)

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0
