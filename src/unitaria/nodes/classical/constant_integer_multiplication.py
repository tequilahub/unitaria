import numpy as np

from ..node import Node
from unitaria.subspace import Subspace

from unitaria.nodes.classical.constant_integer_addition import ConstantIntegerAddition
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.basic.block_diagonal import BlockDiagonal
from unitaria.nodes.permutation.permutation import PermuteRegisters
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.basic.mul import Mul


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

    def __init__(self, *args, bits: int = None, constant: int = None):
        if len(args) > 0 or bits is None or constant is None:
            raise TypeError(
                "ConstantIntegerMultiplication constructor requires bits=... and constant=... as keyword arguments."
            )
        super().__init__(2**bits, 2**bits)
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
            return Identity(subspace=Subspace(bits=1))
        result = None
        for i in reversed(range(self.bits - 1)):
            add_bits = self.bits - 1 - i
            c = (self.constant >> 1) & ((1 << add_bits) - 1)
            const_add = BlockDiagonal(
                Identity(subspace=Subspace(bits=add_bits)),
                ConstantIntegerAddition(bits=add_bits, constant=c),
)
            permutation = PermuteRegisters(Subspace(bits=add_bits + 1), [add_bits] + list(range(add_bits)))
            # TODO: The skip_projection can be removed onces this is done automatically
            const_add = Mul(
                Adjoint(permutation),
                Mul(const_add, permutation, skip_projection=True),
                skip_projection=True,
            )
            const_add = Identity(subspace=Subspace(bits=i)) & const_add
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
