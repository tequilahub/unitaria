from abc import abstractmethod
from typing import Sequence

import numpy as np

from unitaria.nodes.node import Node
from unitaria.subspace import Subspace


class Classical(Node):
    """
    Abstract superclass for nodes performing classical computation.

    :param input_bits:
        The number of bits in the input of the computed function,
        or a Sequence of the number of bits if there are multiple inputs
    :param output_bits:
        The number of bits in the output of the computed function
        or a Sequence of the number of bits if there are multiple outputs
    :raises ValueError: If any of the input_bits or output_bits are less than 1.
    """

    input_bits: int | Sequence[int]
    output_bits: int | Sequence[int]

    def __init__(self, input_bits: int | Sequence[int], output_bits: int | Sequence[int]):
        self.input_bits = input_bits if isinstance(input_bits, Sequence) else [input_bits]
        self.output_bits = output_bits if isinstance(output_bits, Sequence) else [output_bits]
        if any(b < 1 for b in self.input_bits) or any(b < 1 for b in self.output_bits):
            raise ValueError()
        super().__init__(2 ** sum(self.input_bits), 2 ** sum(self.output_bits))

    def _subspace_in(self) -> Subspace:
        return Subspace("0" * max(0, sum(self.output_bits) - sum(self.input_bits)) + "#" * sum(self.input_bits))

    def _subspace_out(self) -> Subspace:
        return Subspace("0" * max(0, sum(self.input_bits) - sum(self.output_bits)) + "#" * sum(self.output_bits))

    def _normalization(self) -> float:
        return 1

    @abstractmethod
    def compute_classical(self, input: np.ndarray | Sequence[np.ndarray]) -> np.ndarray | Sequence[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def compute_reverse_classical(self, input: np.ndarray | Sequence[np.ndarray]) -> np.ndarray | Sequence[np.ndarray]:
        raise NotImplementedError

    def compute(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        output = np.zeros(outer_shape + [2 ** sum(self.output_bits)], dtype=np.complex128)
        output = output.reshape([-1, 2 ** sum(self.output_bits)])
        input = input.reshape([-1, 2 ** sum(self.input_bits)])
        indices = [np.arange(2**i, dtype=np.int32) for i in self.input_bits]
        if len(indices) == 1:
            mapped = self.compute_classical(indices[0])
        else:
            indices = np.meshgrid(*indices, indexing="ij")
            indices = [a.flatten("F") for a in indices]
            mapped = self.compute_classical(indices)
            mapped = sum(mapped[i] * 2 ** (sum(self.output_bits[:i])) for i in range(len(mapped)))

        output[:, mapped] = input
        return output.reshape(outer_shape + [2 ** sum(self.output_bits)])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        output = np.zeros(outer_shape + [2 ** sum(self.output_bits)], dtype=np.complex128)
        output = output.reshape([-1, 2 ** sum(self.output_bits)])
        input = input.reshape([-1, 2 ** sum(self.input_bits)])
        indices = [np.arange(2**i, dtype=np.int32) for i in self.input_bits]
        if len(indices) == 1:
            mapped = self.compute_reverse_classical(indices[0])
        else:
            indices = np.meshgrid(*indices, indexing="ij")
            indices = [a.flatten("F") for a in indices]
            mapped = self.compute_reverse_classical(indices)
            mapped = sum(mapped[i] * 2 ** (sum(self.output_bits[:i])) for i in range(len(mapped)))

        output[:, mapped] = input
        return output.reshape(outer_shape + [2 ** sum(self.output_bits)])
