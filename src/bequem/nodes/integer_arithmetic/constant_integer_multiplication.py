import numpy as np

from ..node import Node
from bequem.subspace import Subspace

from bequem.nodes.integer_arithmetic.constant_integer_addition import ConstantIntegerAddition
from bequem.nodes.proxy_node import ProxyNode
from bequem.nodes.identity import Identity
from bequem.nodes.controlled_operations.block_diagonal import BlockDiagonal
from bequem.nodes.permutation import PermuteRegisters
from bequem.nodes.basic_operations.adjoint import Adjoint
from bequem.nodes.ring_operations.mul import Mul


class ConstantIntegerMultiplication(ProxyNode):
    """
    Node implementing the (wrapping) multiplication of an odd constant with an integer.

    The constraint for the factor to be odd means the multiplication operation is invertible.

    :param bits:
        The size of the quantum state. The addition is performed modulo ``2 ** bits``.
    :param constant:
        The contant factor that should be multiplied. Has to be positive and odd.
    """

    bits: int
    constant: int

    def __init__(self, bits: int, constant: int):
        if constant < 0:
            raise ValueError(f"Constant factor {constant} is negative.")
        if constant & 1 != 1:
            raise ValueError(f"Constant factor {constant} is uneven. This would result in a non-reversible operation.")
        if bits < 1:
            raise ValueError()
        self.bits = bits
        self.constant = constant

    def children(self) -> list[Node]:
        return []

    def parameters(self) -> dict:
        return {"bits": self.bits, "constant": self.constant}

    def definition(self) -> Node:
        if self.bits == 1:
            assert self.constant == 1
            return Identity(Subspace(1))
        result = None
        for i in reversed(range(self.bits - 1)):
            add_bits = self.bits - 1 - i
            c = (self.constant >> 1) & ((1 << add_bits) - 1)
            const_add = BlockDiagonal(Identity(Subspace(add_bits)), ConstantIntegerAddition(add_bits, c))
            permutation = PermuteRegisters(Subspace(add_bits + 1), [add_bits] + list(range(add_bits)))
            # TODO: The skip_projection can be removed onces this is done automatically
            const_add = Mul(
                Adjoint(permutation),
                Mul(const_add, permutation, skip_projection=True),
                skip_projection=True,
            )
            const_add = Identity(Subspace(i)) & const_add
            if result is not None:
                result = Mul(result, const_add, skip_projection=True)
            else:
                result = const_add
        return result

    def compute(self, input: np.ndarray) -> np.ndarray:
        inv = self._mod_inv()
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, 2**self.bits])
        indices = np.array([(i * inv) % 2**self.bits for i in range(2**self.bits)], dtype=np.uint32)
        output = input[:, indices]
        return output.reshape(outer_shape + [-1])

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        outer_shape = list(input.shape[:-1])
        input = input.reshape([-1, 2**self.bits])
        indices = np.array(
            [(i * self.constant) % 2**self.bits for i in range(2**self.bits)],
            dtype=np.uint32,
        )
        output = input[:, indices]
        return output.reshape(outer_shape + [-1])

    def _mod_inv(self):
        return pow(self.constant, -1, mod=2**self.bits)
