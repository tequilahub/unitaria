import numpy as np
import unitaria as ut
import tequila as tq
from unitaria.util import cached_property


class LCG(ut.ProxyNode):
    bits: int
    constant_add: int
    constant_mul: int

    def __init__(self, bits: int, constant_add: int | None = None, constant_mul: int | None = None):
        super().__init__(2 ** (2 * bits), 2 ** (2 * bits))
        self.bits = bits
        if constant_add is None:
            self.constant_add = 1 + np.random.randint(0, 2 ** (bits - 1)) * 2
        else:
            self.constant_add = constant_add
        if constant_mul is None:
            self.constant_mul = 1 + np.random.randint(0, 2 ** (bits - 2)) * 4
        else:
            self.constant_mul = constant_mul

    def definition(self) -> ut.Node:
        acc_mul = self.constant_mul
        acc_add = self.constant_add
        result = None
        for i in range(self.bits):
            # TODO: skip_projection should be done automatically here
            step = ut.ConstantIntegerAddition(self.bits, acc_add) @ ut.ConstantIntegerMultiplication(self.bits, acc_mul)
            controlled = ut.Identity(ut.Subspace("#" * (self.bits - i - 1))) & (
                ut.Identity(ut.Subspace("#" * i + "00" + "#" * self.bits)) | (ut.Identity(ut.Subspace("#" * i)) & step)
            )
            if result is None:
                result = controlled
            else:
                # TODO: skip_projection should be done automatically here
                result = controlled @ result
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
    ut.verify(LCG(2, 1, 1))
    ut.verify(LCG(2, 1, 3))


class XorShift(ut.Classical):
    bits: int
    shift: int

    def __init__(self, bits: int, shift: int):
        super().__init__(bits, bits)
        assert shift < bits
        self.bits = bits
        self.shift = shift

    def compute_classical(self, input: np.ndarray) -> np.ndarray:
        return input ^ (input >> self.shift)

    def compute_reverse_classical(self, input: np.ndarray) -> np.ndarray:
        return input ^ (input >> self.shift)

    def _circuit(self, target, clean_ancillae, borrowed_ancillae) -> ut.Circuit:
        circuit = ut.Circuit()
        circuit.n_qubits = self.bits

        for i in range(self.bits - self.shift):
            circuit += tq.gates.CX(i + self.shift, i)

        return circuit

    def clean_ancilla_count(self) -> int:
        return 0

    def borrowed_ancilla_count(self) -> int:
        return 0


class ControlledRightShift(ut.Classical):
    bits: int

    def __init__(self, bits: int):
        self.bits = bits
        self.log_bits = int(np.log2(bits))
        assert 2**self.log_bits == bits
        super().__init__([bits, self.log_bits], [bits, self.log_bits])

    def compute_classical(self, input: list[np.ndarray]) -> np.ndarray:
        x, shift = input
        x = (x >> shift) | ((x << (self.bits - shift)) & ((1 << self.bits) - 1))
        return x, shift

    def compute_reverse_classical(self, input: list[np.ndarray]) -> np.ndarray:
        x, shift = input
        x = (x >> (self.bits - shift)) | ((x << shift) & ((1 << self.bits) - 1))
        return x, shift

    @cached_property
    def _increments(self):
        return [ut.Increment(bits=b) for b in range(1, self.log_bits + 1)]

    def _circuit_internal(self, circuit, control, target, log_bits, ancillae) -> ut.Circuit:
        assert len(target) == 2**log_bits
        if log_bits == 1:
            circuit += tq.gates.SWAP(target[0], target[1], control=control[0])
        else:
            for i in range(0, 2 ** (log_bits - 1), 2):
                circuit += tq.gates.SWAP(target[i], target[i + 1], control=control[0])
            self._circuit_internal(circuit, control[1:], target[::2], log_bits - 1, ancillae)

            circuit += self._increments[log_bits - 1].circuit(control, ancillae, [])

            self._circuit_internal(circuit, control[1:], target[1::2], log_bits - 1, ancillae)

            circuit += self._increments[log_bits - 1].circuit(control, ancillae, []).adjoint()

    def _circuit(self, target, clean_ancillae, borrowed_ancillae) -> ut.Circuit:
        circuit = ut.Circuit()
        circuit.n_qubits = self.bits + self.log_bits
        self._circuit_internal(circuit, target[self.bits :], target[: self.bits], self.log_bits, clean_ancillae)

        return circuit

    def clean_ancilla_count(self) -> int:
        return max([node.clean_ancilla_count() for node in self._increments])

    def borrowed_ancilla_count(self) -> int:
        return 0


class PCG6to4(ut.ProxyNode):
    def __init__(self, constant_add: int, constant_mul: int):
        super().__init__(2**12, 2**10)
        self.constant_add = constant_add
        self.constant_mul = constant_mul

    def definition(self):
        return (ut.Identity(ut.Subspace("######")) & (ControlledRightShift(4) @ XorShift(6, 2))) @ LCG(
            6, self.constant_add, self.constant_mul
        )


def test_pcg():
    pcg = PCG6to4(5, 17)

    seed = 42

    N = 2**6
    for i in range(N):
        input = np.zeros((N, N))
        input[i, seed] = 1
        input = input.reshape(-1)
        output = pcg.compute(input).reshape((N, N))
        output = np.argmax(output[i]) % (2**4)
        print(f"PCG({i}) = {output}")


if __name__ == "__main__":
    test_pcg()
