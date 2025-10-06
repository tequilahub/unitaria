from typing import Sequence

import numpy as np

from unitaria import Node, Subspace, Circuit
from unitaria.circuits.qft import qft_circuit


class FourierTransform(Node):
    """
    Node representing a discrete Fourier transform
    ``F(x)_k = sum_(n = 0)^(2^bits - 1) x_n exp(-i 2 pi k n / 2^bits)``
    Note that this uses the DFT convention with the minus sign in the
    exponent, which differs from the usual convention for the QFT.

    :param bits:
        The number of bits that this Node acts on.
    """

    bits: int

    def __init__(self, bits: int):
        super().__init__(2**bits, 2**bits)
        if bits < 1:
            raise ValueError()

        self.bits = bits

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"bits": self.bits}

    def _subspace_in(self) -> Subspace:
        return Subspace(self.bits)

    def _subspace_out(self) -> Subspace:
        return Subspace(self.bits)

    def _normalization(self) -> float:
        return 1

    def compute(self, input: np.ndarray) -> np.ndarray:
        return np.fft.fft(input) / np.sqrt(2**self.bits)

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return np.fft.ifft(input) * np.sqrt(2**self.bits)

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        # dagger() because the QFT uses different exponent sign conventions than the DFT
        return Circuit(tq_circuit=qft_circuit(self.bits).dagger()).map_qubits({i: target[i] for i in range(self.bits)})

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0
