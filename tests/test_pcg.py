import numpy as np
from unitaria.nodes import ProxyNode, Node, ConstantIntegerAddition, ConstantIntegerMultiplication, Identity, Mul
from unitaria.subspace import Subspace, ID, ZeroQubit
from unitaria.verifier import verify


class LCG(ProxyNode):
    bits: int
    constant_add: int
    constant_mul: int

    def __init__(self, bits: int, constant_add: int | None = None, constant_mul: int | None = None):
        self.bits = bits
        if constant_add is None:
            self.constant_add = 1 + np.random.randint(0, 2 ** (bits - 1)) * 2
        else:
            self.constant_add = constant_add
        if constant_mul is None:
            self.constant_mul = 1 + np.random.randint(0, 2 ** (bits - 2)) * 4
        else:
            self.constant_mul = constant_mul

    def definition(self) -> Node:
        acc_mul = self.constant_mul
        acc_add = self.constant_add
        result = None
        for i in range(self.bits):
            # TODO: skip_projection should be done automatically here
            step = Mul(
                ConstantIntegerMultiplication(self.bits, acc_mul),
                ConstantIntegerAddition(self.bits, acc_add),
                skip_projection=True,
            )
            controlled = (
                Identity(Subspace([ID] * self.bits + [ZeroQubit()] * 2 + [ID] * i)) | (step & Identity(i))
            ) & Identity(self.bits - i - 1)
            if result is None:
                result = controlled
            else:
                # TODO: skip_projection should be done automatically here
                result = Mul(result, controlled, skip_projection=True)
            acc_add = (acc_add + acc_mul * acc_add) % (2**self.bits)
            acc_mul = (acc_mul**2) % (2**self.bits)

        return result

    def compute(self, input):
        inv_mul = pow(self.constant_mul, -1, mod=2**self.bits)
        inv_add = 2**self.bits - self.constant_add
        outer_shape = list(input.shape[:-1])
        output = input.reshape([-1, 2**self.bits, 2**self.bits]).copy()
        acc = np.arange(2**self.bits)
        for i in range(2**self.bits):
            output[:, i, :] = output[:, i, acc]
            acc = (inv_mul * (acc + inv_add)) % 2**self.bits
        return output.reshape(outer_shape + [-1])

    def compute_adjoint(self, input):
        outer_shape = list(input.shape[:-1])
        output = input.reshape([-1, 2**self.bits, 2**self.bits]).copy()
        acc = np.arange(2**self.bits)
        for i in range(2**self.bits):
            output[:, i, :] = output[:, i, acc]
            acc = (self.constant_mul * acc + self.constant_add) % 2**self.bits
        return output.reshape(outer_shape + [-1])


def test_lcg():
    verify(LCG(2, 1, 1))
    verify(LCG(2, 1, 3))
