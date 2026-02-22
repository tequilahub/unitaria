from abc import abstractmethod

import numpy as np

from ..node import Node
from unitaria.subspace import Subspace


class Classical(Node):
    """
    Abstract superclass for nodes performing classical computation.

    :param input_bits:
        The number of bits in the input of the computed function
    :param output_bits:
        The number of bits in the output of the computed function
    """

    input_bits: int
    output_bits: int

    def __init__(self, input_bits: int, output_bits):
        super().__init__(2**input_bits, 2**output_bits)
        if input_bits < 1 or output_bits < 1:
            raise ValueError()
        self.input_bits = input_bits
        self.output_bits = output_bits

    def _subspace_in(self) -> Subspace:
        return Subspace(registers=self.input_bits, zero_qubits=max(0, self.output_bits - self.input_bits))

    def _subspace_out(self) -> Subspace:
        return Subspace(registers=self.output_bits, zero_qubits=max(0, self.input_bits - self.output_bits))

    def _normalization(self) -> float:
        return 1

    @abstractmethod
    def compute_classical(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_reverse_classical(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def compute(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        output = np.zeros(outer_shape + [2**self.output_bits], dtype=np.complex128)
        output = output.reshape([-1, 2**self.output_bits])
        input = input.reshape([-1, 2**self.input_bits])
        indices = np.arange(2**self.input_bits, dtype=np.int32)
        output[:, self.compute_classical(indices)] = input[:, indices]
        return output.reshape(outer_shape + [2**self.output_bits])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        output = np.zeros(outer_shape + [2**self.input_bits], dtype=np.complex128)
        output = output.reshape([-1, 2**self.input_bits])
        input = input.reshape([-1, 2**self.output_bits])
        indices = np.arange(2**self.output_bits, dtype=np.int32)
        output[:, self.compute_reverse_classical(indices)] = input[:, indices]
        return output.reshape(outer_shape + [2**self.input_bits])
