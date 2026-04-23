"""
Objects for specifying a subspace of a quantum statespace.
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import tequila as tq

from unitaria.circuit import Circuit
from unitaria.util import cached_property


class Subspace:
    """
    Subspace of the statespace of a number of qubits.

    In ``unitaria``, subspaces are always the span of vectors in the
    computational basis, so this object just stores the indices of
    these computational basis states.

    Constructors
    ------------

    If you just need a subspace with a given dimension, use `Subspace.from_dim`.

    Otherwise, the preferred way to construct subspaces is using the string
    constructor. It takes a string consisting of ``#`` and ``0``, respectively
    representing a bit that can be in either state, or a bit that should be in
    its ``0`` state.

        >>> import unitaria as ut
        >>> ut.Subspace("#").enumerate_basis()
        array([0, 1], dtype=int32)
        >>> ut.Subspace("0").enumerate_basis()
        array([0], dtype=int32)

    Alternatively, the subspace can be constructed according to its internal
    representation, see below.

    When given multiple symbols, they are combined using tensor products as
    expected.

        >>> import unitaria as ut
        >>> ut.Subspace("##0").enumerate_basis()
        array([0, 2, 4, 6], dtype=int32)

    Subspaces can be combined using the operators ``&`` (tensor product) or
    ``|`` direct sum.

        >>> import unitaria as ut
        >>> (ut.Subspace("#") & ut.Subspace("0")).enumerate_basis()
        array([0, 2], dtype=int32)
        >>> (ut.Subspace("#") | ut.Subspace("0")).enumerate_basis()
        array([0, 1, 2], dtype=int32)

    Intutively, with the ``|`` operator the highest bit decides the subspace
    of the lower bits. So in the example above, when the highes bit is ``0``
    then the lower bit can either be ``0`` or ``1``, but when the highest bit
    is ``1`` then the lower bit has to be ``0``.

    Reading the subspace
    --------------------

    The method `enumerate_basis` gives a list of all the indices in the
    subspace, but this should typically only be used for verification. It
    does not make sense as part of an efficient quantum algorithm, since the
    complexity of `enumerate_basis` is always linear in the dimension.

    The methods `test_basis` on the other hand is efficient. It checks whether a
    given index is in the subspace.

    The dimension of the subspace can be obtained using `Subspace.dimension`.
    The dimension of the super-space can be caluclated from the number of
    qubits, which is stored in `Subspace.total_qubits`. Specifically, the
    super-space dimension is ``2 ** subspace.total_qubits``.

        >>> import unitaria as ut
        >>> ut.Subspace("0#0").dimension
        2
        >>> ut.Subspace("0#0").total_qubits
        3

    Internal representation
    -----------------------

    Typically one does not need to inspect `Subspace` objects, most properties
    can be derived using its methods. Internally, the object stores a
    decomposition of the subspace into tensor factors. This decomposition is
    simplified at construction and so does not have to correspond to the factors
    that are put in. Any of the factors can be either

    * a `ZeroQubitSubspace`, indicating the subspace of the space of one
      qubit, where this qubit is zero, or
    * a `ControlledSubspace`, indicating that a subspace, where the
      most siginificant bit determines the subspaces of the lower bits.
      These subspaces are given by `ControlledSubspace.case_zero` and
      `ControlledSubspace.case_one`. With this one can also construct the full
      subspace of a qubit by `ControlledSubspace(Subspace(), Subspace())`.
      (For this there is also the constant `~unitaria.FullQubitSubspace`.)

    :param tensor_factors: List or string of factors making up the subspace, see below.

    :raises ValueError:
        If a string with characters other than ``#`` or ``0`` is given or a list
        with anything that is not a `SubspaceFactor`
    """

    tensor_factors: list[SubspaceFactor]

    def __init__(self, tensor_factors: list[SubspaceFactor] | str = []):
        if isinstance(tensor_factors, str):
            self.tensor_factors = []
            for c in reversed(tensor_factors):
                if c == "0":
                    self.tensor_factors.append(ZeroQubitSubspace())
                elif c == "#":
                    self.tensor_factors.append(FullQubitSubspace)
                else:
                    raise ValueError()
        else:
            self.tensor_factors = tensor_factors

        for factor in self.tensor_factors:
            if not isinstance(factor, SubspaceFactor):
                if factor is ZeroQubitSubspace:
                    raise ValueError(
                        f"{factor} is not a valid factor in a Subspace. Use `ZeroQubit()` instead of `ZeroQubit`"
                    )
                raise ValueError(f"{factor} is not a valid factor in a Subspace.")
        simplified_factors = []
        for factor in self.tensor_factors:
            if isinstance(factor, ControlledSubspace):
                simplified_factors += factor.simplify()
            else:
                simplified_factors.append(factor)
        self.tensor_factors = simplified_factors

    @staticmethod
    def from_dim(dim: int, bits: int | None = None) -> Subspace:
        if bits is None:
            bits = int(np.ceil(np.log2(dim)))
        if dim == 1:
            return Subspace("0" * bits)
        min_bits = int(np.ceil(np.log2(dim)))
        case_zero = Subspace("#" * (min_bits - 1))
        case_one = Subspace.from_dim(dim - 2 ** (min_bits - 1), bits=min_bits - 1)
        return (case_zero | case_one) & Subspace("0" * (bits - min_bits))

    def __repr__(self) -> str:
        if len(self.tensor_factors) == 0:
            return "Subspace()"
        string_constructor = ""
        output = ""
        for factor in self.tensor_factors:
            if factor == ZeroQubitSubspace():
                string_constructor = "0" + string_constructor
            elif factor == FullQubitSubspace:
                string_constructor = "#" + string_constructor
            elif isinstance(factor, ControlledSubspace):
                if len(string_constructor) != 0:
                    if len(output) == 0:
                        output = f'Subspace("{string_constructor}")'
                    else:
                        output = f'Subspace("{string_constructor}") & {output}'
                if len(output) == 0:
                    output = f"({repr(factor.case_zero)} | {repr(factor.case_one)})"
                else:
                    output = f"({repr(factor.case_zero)} | {repr(factor.case_one)}) & {output}"
            else:
                raise NotImplementedError
        if len(string_constructor) != 0:
            if len(output) == 0:
                output = f'Subspace("{string_constructor}")'
            else:
                output = f'Subspace("{string_constructor}") & {output}'
        return output

    @cached_property
    def dimension(self) -> int:
        """
        The dimension of the subspace
        """
        dimension = 1
        for factor in self.tensor_factors:
            dimension *= factor.dimension()

        return dimension

    @cached_property
    def total_qubits(self) -> int:
        """
        The number of qubits of the state space in which the subspace lives

        The dimension of the state space is ``2 ** total_qubits``
        """
        total_qubits = 0
        for factor in self.tensor_factors:
            total_qubits += factor.total_qubits()
        return total_qubits

    def __eq__(self, other) -> bool:
        return self.tensor_factors == other.tensor_factors

    def match_nonzero(self, other: Subspace) -> bool:
        return self.nonzero_factors() == other.nonzero_factors()

    def __str__(self) -> str:
        if len(self.tensor_factors) == 0:
            return "<zero qubit subspace>"
        tree = self.draw_tree()
        output = ""
        lines = tree.splitlines()
        max_depth = (len(lines) + 1) // 2 - 1
        digits = int(np.ceil(np.log10(max(max_depth, 1) + 1)))
        for i, line in enumerate(lines):
            if i % 2 == 1:
                output += ("{:0" + str(digits) + "} ").format(max_depth - i // 2)
            else:
                output += " " * (digits + 1)
            output += line
            if i != len(lines) - 1:
                output += "\n"
        return output

    def draw_tree(self) -> str:
        output = ""
        for i, factor in enumerate(reversed(self.tensor_factors)):
            if i == 0:
                output += "│\n"
            str_factor = str(factor)
            output += str_factor
            output += "\n"
            if i != len(self.tensor_factors) - 1:
                last_line = str_factor.rsplit("\n", 1)
                if len(last_line) > 1:
                    last_line = last_line[-1]
                    for i, c in enumerate(last_line):
                        if i == 0:
                            output += "╔"
                        elif i == len(last_line) - 1:
                            if c == "0":
                                output += "╛"
                            else:
                                output += "╝"
                        elif c == " ":
                            output += "═"
                        elif c == "0":
                            output += "╧"
                        else:
                            output += "╩"
                    output += "\n"
                else:
                    if str_factor == "0":
                        output += "│\n"
                    else:
                        output += "║\n"
        return output

    def initial_zeros(self) -> int:
        for i in reversed(range(len(self.tensor_factors))):
            if not isinstance(self.tensor_factors[i], ZeroQubitSubspace):
                return len(self.tensor_factors) - i - 1
        return len(self.tensor_factors)

    def nonzero_factors(self) -> list[SubspaceFactor]:
        initial_zeros = self.initial_zeros()
        if initial_zeros == 0:
            return self.tensor_factors
        else:
            return self.tensor_factors[:-initial_zeros]

    def test_basis(self, bits: int) -> bool:
        """
        Tests whether the given basis state is inside the subspace
        """
        if bits >= 2**self.total_qubits:
            raise ValueError
        for factor in self.tensor_factors:
            match factor:
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
                case ZeroQubitSubspace():
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

    def circuit(self, target: Sequence[int], flag: int, ancillae: Sequence[int]) -> Circuit:
        """
        A circuit which checks whether a state is inside the subspace.

        The result of the check will be stored in a flag qubit where
        the index is the output of ``total_qubits``.
        Specifically, this qubit will be flipped if the other qubits
        represent a state outside the embedded vector space.
        """
        # Index of the current factor
        index = 0
        # Next unused ancilla index
        ancilla_index = 0
        # Set of intermediate flags, if one of them is set, the result flag will be set
        intermediate_flags = []

        circuit = Circuit()

        for factor in self.tensor_factors:
            if isinstance(factor, ZeroQubitSubspace):
                # Zero qubits already behave like flag qubits, so we only need to toggle it
                circuit += tq.gates.X(target=target[index])
                intermediate_flags.append(target[index])
                index += 1
                continue

            if factor == FullQubitSubspace:
                # No need to do anything here
                index += 1
                continue

            if not isinstance(factor, ControlledSubspace):
                raise ValueError("SubspaceFactors that aren't of type Qubit are unsupported")

            circuit += factor.circuit(
                target=target[index : index + factor.total_qubits()],
                flag=ancillae[ancilla_index],
                ancillae=ancillae[ancilla_index + 1 :],
            )
            circuit += tq.gates.X(target=ancillae[ancilla_index])
            intermediate_flags.append(ancillae[ancilla_index])

            index += factor.total_qubits()
            ancilla_index += 1

        # Apply the constructed circuit, add a multi-controlled NOT to determine
        # if the result flag is set, then uncompute the rest of the circuit
        circuit = (
            circuit + tq.gates.X(target=flag, control=intermediate_flags) + tq.gates.X(target=flag) + circuit.adjoint()
        )

        return circuit

    def clean_ancilla_count(self) -> int:
        controlled_subspaces = filter(
            lambda r: isinstance(r, ControlledSubspace) and r != FullQubitSubspace, self.tensor_factors
        )
        return max(
            [i + r.clean_ancilla_count() for (i, r) in enumerate(controlled_subspaces, start=1)],
            default=0,
        )

    def verify_circuit(self):
        circuit = self.circuit(
            range(self.total_qubits),
            self.total_qubits,
            range(
                self.total_qubits + 1,
                self.total_qubits + 1 + self.clean_ancilla_count(),
            ),
        )
        circuit.n_qubits = self.total_qubits + 1 + self.clean_ancilla_count()
        for i in range(2**self.total_qubits):
            result = circuit.simulate(i)
            if self.test_basis(i):
                assert result[i] == 1
            else:
                assert result[i + 2**self.total_qubits] == 1

    def case_zero(self) -> Subspace:
        initial_zeros = self.initial_zeros()
        if initial_zeros == len(self.tensor_factors):
            return None

        return Subspace(
            self.tensor_factors[: -(initial_zeros + 1)]
            + self.tensor_factors[-(initial_zeros + 1)].case_zero.tensor_factors,
        )

    def case_one(self) -> Subspace:
        initial_zeros = self.initial_zeros()
        if initial_zeros == len(self.tensor_factors):
            return None

        return Subspace(
            self.tensor_factors[: -(initial_zeros + 1)]
            + self.tensor_factors[-(initial_zeros + 1)].case_one.tensor_factors,
        )

    def __and__(self, other: Subspace) -> Subspace:
        return Subspace(other.tensor_factors + self.tensor_factors)

    def __or__(self, other: Subspace) -> Subspace:
        return Subspace([ControlledSubspace(self, other)])


class SubspaceFactor(ABC):
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
class ZeroQubitSubspace(SubspaceFactor):
    def total_qubits(self) -> int:
        return 1

    def dimension(self) -> int:
        return 1

    def __str__(self) -> str:
        return "0"


@dataclass(frozen=True)
class ControlledSubspace(SubspaceFactor):
    """
    SubspaceFactor where the most significant qubit determines the subspace
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

    def total_qubits(self) -> int:
        return 1 + self.case_zero.total_qubits

    def dimension(self) -> int:
        return self.case_zero.dimension + self.case_one.dimension

    def __str__(self) -> str:
        tree0 = self.case_zero.draw_tree()
        tree1 = self.case_one.draw_tree()
        lines0 = tree0.splitlines()
        lines1 = tree1.splitlines()

        assert len(lines0) == len(lines1), str(lines0) + "," + str(lines1)

        if len(lines0) == 0:
            return "#"

        w0 = max(map(len, lines0))
        w1 = max(map(len, lines1))

        output = "?─┬" + "─" * w0 + "┐\n"
        for i in range(len(lines0)):
            output += "  "
            output += lines0[i]
            output += " " * (w0 - len(lines0[i]) + 1)
            output += lines1[i]
            output += " " * (w1 - len(lines1[i]))
            if i != len(lines0) - 1:
                output += "\n"

        return output

    def simplify(self) -> list[SubspaceFactor]:
        """
        Returns a potentially simpler representation of this factor

        Specifically, if `case_one` and `case_zero` agree in a number of lowest
        qubits, this common part can be factored out. E.g.

            >>> import unitaria as ut
            >>> ut.ControlledSubspace(ut.Subspace("##"), ut.Subspace("0#")).simplify()
            [ControlledSubspace(case_zero=Subspace(), case_one=Subspace()), ControlledSubspace(case_zero=Subspace("#"), case_one=Subspace("0"))]
        """
        min_len = min(len(self.case_zero.tensor_factors), len(self.case_one.tensor_factors))
        for i in range(min_len):
            if self.case_zero.tensor_factors[i] != self.case_one.tensor_factors[i]:
                return self.case_zero.tensor_factors[:i] + [
                    ControlledSubspace(
                        Subspace(tensor_factors=self.case_zero.tensor_factors[i:]),
                        Subspace(tensor_factors=self.case_one.tensor_factors[i:]),
                    )
                ]
        return self.case_zero.tensor_factors[:min_len] + [
            ControlledSubspace(
                Subspace(tensor_factors=self.case_zero.tensor_factors[min_len:]),
                Subspace(tensor_factors=self.case_one.tensor_factors[min_len:]),
            )
        ]

    def circuit(self, target: Sequence[int], flag: int, ancillae: Sequence[int]) -> Circuit:
        """
        A circuit which checks whether a state is inside the subspace.

        The result of the check will be stored in the flag qubit.
        Specifically, this qubit will be flipped if the other qubits
        represent a state outside the embedded vector space.
        """
        circuit = Circuit()
        control = target[-1]
        circuit_zero = self.case_zero.circuit(target=target[:-1], flag=flag, ancillae=ancillae)
        circuit_one = self.case_one.circuit(target=target[:-1], flag=flag, ancillae=ancillae)
        circuit += tq.gates.X(target=control)
        circuit += circuit_zero.add_controls(control)
        circuit += tq.gates.X(target=control)
        circuit += circuit_one.add_controls(control)
        return circuit

    def clean_ancilla_count(self) -> int:
        return max(self.case_zero.clean_ancilla_count(), self.case_one.clean_ancilla_count())


FullQubitSubspace = ControlledSubspace(Subspace([]), Subspace([]))
