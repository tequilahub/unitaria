"""
Objects for specifying a subspace of a quantum statespace.
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from unitaria.circuit import Circuit
import tequila as tq


class Subspace:
    # TODO: More documentation
    """
    Subspace of the statespace of a number of qubits.

    :param registers:
        List of registers, each describing a subspace. The total subspace of
        all registers is the tensor product of the individual subspaces.
    :param zero_qubits:
        Add an additional ``zero_qubits * [ZeroQubit()]`` to ``registers``.
    """

    registers: list[Register]

    def __init__(self, registers: list[Register] | int, zero_qubits: int = 0):
        if isinstance(registers, (int, np.integer)):
            registers = [ID] * registers
        for register in registers:
            if not isinstance(register, Register):
                raise ValueError(f"{register} is not a valid register in a Subspace")
        simplified_registers = []
        for register in registers:
            if isinstance(register, ControlledSubspace):
                simplified_registers += register.simplify()
            else:
                simplified_registers.append(register)
        self.registers = simplified_registers + [ZeroQubit()] * zero_qubits
        self._dimension = None
        self._total_qubits = None

    @property
    def dimension(self) -> int:
        """
        The dimension of the subspace
        """
        if self._dimension is None:
            self._dimension = 1
            for register in self.registers:
                self._dimension *= register.dimension()

        return self._dimension

    @property
    def total_qubits(self) -> int:
        """
        The number of qubits of the state space in which the subspace lives

        The dimension of the state space is ``2 ** total_qubits``
        """
        if self._total_qubits is None:
            self._total_qubits = 0
            for register in self.registers:
                self._total_qubits += register.total_qubits()
        return self._total_qubits

    def __eq__(self, other) -> bool:
        return self.registers == other.registers

    def match_nonzero(self, other: Subspace) -> bool:
        return self.nonzero_registers() == other.nonzero_registers()

    def __repr__(self) -> str:
        trailing_zeros = self.trailing_zeros()
        registers = self.nonzero_registers()
        str_registers = str(len(registers))
        for register in registers:
            if register != ID:
                str_registers = str(registers)
                break
        if trailing_zeros == 0:
            return f"Subspace({str_registers})"
        return f"Subspace({str_registers}, zero_qubits={trailing_zeros})"

    def is_trivial(self) -> bool:
        """
        Tests whether the subspace only contains the ``|0>`` state
        """
        return self.trailing_zeros() == len(self.registers)

    def trailing_zeros(self) -> int:
        for i in reversed(range(len(self.registers))):
            if not isinstance(self.registers[i], ZeroQubit):
                return len(self.registers) - i - 1
        return len(self.registers)

    def nonzero_registers(self) -> list[Register]:
        trailing_zeros = self.trailing_zeros()
        if trailing_zeros == 0:
            return self.registers
        else:
            return self.registers[:-trailing_zeros]

    def test_basis(self, bits: int) -> bool:
        """
        Tests whether the given basis state is inside the subspace
        """
        if bits >= 2**self.total_qubits:
            raise ValueError
        for register in self.registers:
            match register:
                case ControlledSubspace(case_zero, case_one):
                    num_qubits = case_zero.total_qubits
                    relevant_bits = bits & ((1 << num_qubits) - 1)
                    control_bit = (bits >> num_qubits) & 1
                    result = None
                    if control_bit == 0:
                        result = case_zero.test_basis(relevant_bits)
                    else:
                        result = case_one.test_basis(relevant_bits)
                    if not result:
                        return False
                    bits = bits >> (num_qubits + 1)
                case ZeroQubit():
                    if bits & 1 != 0:
                        return False
                    bits = bits >> 1
                case _:
                    raise NotImplementedError
        return True

    def enumerate_basis(self) -> np.ndarray:
        """
        Enumerates the basis states inside the subspace
        """
        return np.fromiter(filter(self.test_basis, range(2**self.total_qubits)), dtype=np.int32)

    def project(self, vector: np.ndarray) -> np.ndarray:
        """
        Projects a state vector the embedding
        """
        return vector[self.enumerate_basis()]

    def circuit(self) -> Circuit:
        """
        A circuit which checks whether a state is inside the subspace.

        The result of the check will be stored in a flag qubit where
        the index is the output of ``total_qubits``.
        Specifically, this qubit will be flipped if the other qubits
        represent a state outside the embedded vector space.
        """
        # Index of the result flag qubit
        result = self.total_qubits
        # Next free qubit that can be used as a flag for one of the registers
        next_flag = self.total_qubits + 1
        # Offset to keep track of the index of the current register
        offset = 0
        # Set of register flags, if one of them is set, the result flag will be set.
        # This is achieved by toggling them and then adding a multi-controlled NOT.
        flags = []

        circuit = Circuit()

        for register in self.registers:
            if isinstance(register, ZeroQubit):
                # Zero qubits already behave like flag qubits, so we only need to toggle it
                circuit += tq.gates.X(target=offset)
                flags.append(offset)
                offset += 1
                continue

            if register == ID:
                # No need to do anything here
                offset += 1
                continue

            if not isinstance(register, ControlledSubspace):
                raise ValueError("Registers that aren't of type Qubit are unsupported")

            register_circuit = register.circuit()
            register_ancillae = register_circuit.n_qubits - register.total_qubits()
            # Main qubits get shifted by the offset
            register_map = {i: i + offset for i in range(register.total_qubits())}
            # Ancilla qubits get shifted to the flag qubit and following qubits
            register_map |= {
                j: j - register.total_qubits() + next_flag
                for j in range(register.total_qubits(), register.total_qubits() + register_ancillae)
            }
            circuit += register_circuit.map_qubits(register_map)
            circuit += tq.gates.X(target=next_flag)
            flags.append(next_flag)

            offset += register.total_qubits()
            next_flag += 1

        # Apply the constructed circuit, add a multi-controlled NOT to determine
        # if the result flag is set, then uncompute the rest of the circuit
        circuit = circuit + tq.gates.X(target=result, control=flags) + tq.gates.X(target=result) + circuit.adjoint()

        return circuit

    def verify_circuit(self):
        circuit = self.circuit()
        for i in range(2**self.total_qubits):
            result = circuit.simulate(i)
            if self.test_basis(i):
                assert result[i] == 1
            else:
                assert result[i + 2**self.total_qubits] == 1

    def case_zero(self) -> Subspace:
        trailing_zeros = self.trailing_zeros()
        assert isinstance(self.registers[-(trailing_zeros + 1)], ControlledSubspace)

        return Subspace(
            self.registers[: -(trailing_zeros + 1)] + self.registers[-(trailing_zeros + 1)].case_zero.registers,
        )

    def case_one(self) -> Subspace:
        trailing_zeros = self.trailing_zeros()
        assert isinstance(self.registers[-(trailing_zeros + 1)], ControlledSubspace)

        return Subspace(
            self.registers[: -(trailing_zeros + 1)] + self.registers[-(trailing_zeros + 1)].case_one.registers,
        )

    @staticmethod
    def from_dim(dim: int, bits: int | None = None) -> Subspace:
        if bits is None:
            bits = int(np.ceil(np.log2(dim)))
        if dim == 1:
            return Subspace(0, bits)
        min_bits = int(np.ceil(np.log2(dim)))
        case_zero = Subspace(min_bits - 1)
        case_one = Subspace.from_dim(dim - 2 ** (min_bits - 1), min_bits - 1)
        return Subspace([ControlledSubspace(case_zero, case_one)], bits - min_bits)


class Register(ABC):
    @abstractmethod
    def total_qubits(self) -> int:
        """
        The number of qubits of the state space in which the subspace lives

        The dimension of the state space is ``2 ** total_qubits``
        """
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """
        The dimension of the subspace
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ZeroQubit(Register):
    def total_qubits(self) -> int:
        return 1

    def dimension(self) -> int:
        return 1


@dataclass(frozen=True, repr=False)
class ControlledSubspace(Register):
    """
    Register where the most significant qubit determines the subspace
    corresponding to lower qubits

    :param case_zero:
        Embedding of lower qubits, if highest qubit is ``|0>``
    :param case_one:
        Embedding of lower qubits, if highest qubit is ``|1>``
    """

    case_zero: Subspace
    case_one: Subspace

    def __post_init__(self):
        assert self.case_zero.total_qubits == self.case_one.total_qubits

    def __repr__(self) -> str:
        if self == ID:
            return "ID"
        return f"ControlledSubspace(case_zero={self.case_zero}, case_one={self.case_one})"

    def total_qubits(self) -> int:
        return 1 + self.case_zero.total_qubits

    def dimension(self) -> int:
        return self.case_zero.dimension + self.case_one.dimension

    def simplify(self) -> list[Register]:
        """
        Returns a potentially simpler representation of this register

        Specifically, if `case_one` and `case_zero` agree in a number of lowest
        qubits, this common part can be factored out. E.g.

            >>> from unitaria.subspace import Subspace, ControlledSubspace
            >>> ControlledSubspace(Subspace(2), Subspace(1, 1)).simplify()
            [ID, ControlledSubspace(case_zero=Subspace(1), case_one=Subspace(0, zero_qubits=1))]
        """
        min_len = min(len(self.case_zero.registers), len(self.case_one.registers))
        for i in range(min_len):
            if self.case_zero.registers[i] != self.case_one.registers[i]:
                return self.case_zero.registers[:i] + [
                    ControlledSubspace(
                        Subspace(self.case_zero.registers[i:]),
                        Subspace(self.case_one.registers[i:]),
                    )
                ]
        return self.case_zero.registers[:min_len] + [
            ControlledSubspace(
                Subspace(self.case_zero.registers[min_len:]),
                Subspace(self.case_one.registers[min_len:]),
            )
        ]

    def circuit(self) -> Circuit:
        """
        A circuit which checks whether a state is inside the subspace.

        The result of the check will be stored in a flag qubit where
        the index is the output of ``total_qubits``.
        Specifically, this qubit will be flipped if the other qubits
        represent a state outside the embedded vector space.
        """
        circuit = Circuit()
        control = self.total_qubits() - 1
        circuit_zero = self.case_zero.circuit()
        circuit_one = self.case_one.circuit()
        max_qubits = max(circuit_zero.n_qubits, circuit_one.n_qubits)
        shift_map = {i: i for i in range(control)}
        # Qubits after the control qubit are shifted by one so they don't overlap with the control
        shift_map |= {j: j + 1 for j in range(control, max_qubits)}
        circuit += tq.gates.X(target=control)
        circuit += circuit_zero.map_qubits(shift_map).add_controls(control)
        circuit += tq.gates.X(target=control)
        circuit += circuit_one.map_qubits(shift_map).add_controls(control)
        return circuit


ID = ControlledSubspace(Subspace([]), Subspace([]))
