import copy
from typing import Sequence

import tequila as tq


def addition_circuit(source: Sequence[int], target: Sequence[int]) -> tq.QCircuit:
    """
    Adds the source register to the target register.
    Reference: https://arxiv.org/abs/0910.2530v1

    :param source: Indices of the source qubits in LSB ordering.
    Requires len(source) >= 2.
    :param target: Indices of the target qubits in LSB ordering.
    Requires len(target) >= len(source).
    :return: A circuit implementing the addition.
    """
    assert 2 <= len(source) <= len(target)
    assert set(source).isdisjoint(set(target))
    n = len(source)

    U = tq.QCircuit()

    for i in range(1, n):
        U += tq.gates.CNOT(control=source[i], target=target[i])

    if len(target) == n + 1:
        U += tq.gates.CNOT(control=source[n - 1], target=target[n])
    elif len(target) > n + 1:
        U += increment_circuit_single_ancilla(target=[source[n - 1]] + list(target[n:]), ancilla=source[0])
        U += tq.gates.X(target=source[n - 1])
        for i in range(n, len(target)):
            U += tq.gates.CNOT(control=source[n - 1], target=target[i])

    for i in reversed(range(1, n - 1)):
        U += tq.gates.CNOT(control=source[i], target=source[i + 1])
    for i in range(n - 1):
        U += tq.gates.Toffoli(first=source[i], second=target[i], target=source[i + 1])

    if len(target) == n + 1:
        U += tq.gates.Toffoli(first=target[n - 1], second=source[n - 1], target=target[n])
    elif len(target) > n + 1:
        U += increment_circuit_single_ancilla(
            target=[target[n - 1], source[n - 1]] + list(target[n:]), ancilla=source[0]
        )
        U += tq.gates.X(target=target[n - 1])
        U += tq.gates.CNOT(control=target[n - 1], target=source[n - 1])

    for i in reversed(range(1, n)):
        U += tq.gates.CNOT(control=source[i], target=target[i])
        U += tq.gates.Toffoli(first=source[i - 1], second=target[i - 1], target=source[i])
    for i in range(1, n - 1):
        U += tq.gates.CNOT(control=source[i], target=source[i + 1])
    for i in range(n):
        U += tq.gates.CNOT(control=source[i], target=target[i])

    if len(target) > n + 1:
        for i in range(n, len(target)):
            U += tq.gates.CNOT(control=source[n - 1], target=target[i])

    return U


def increment_circuit_single_ancilla(target: Sequence[int], ancilla: int) -> tq.QCircuit:
    """
    Increments the target register.
    Reference: https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html

    :param target: Indices of the target qubits in LSB ordering.
    :param ancilla: Index of the ancilla qubit.
    Can be in any state, and will be returned to that state by the end of the circuit.
    :return: A circuit implementing the increment.
    """
    assert ancilla not in target

    n = len(target)
    split = (n + 1) // 2

    U = tq.QCircuit()

    U += increment_circuit_n_ancillae(target=[ancilla] + list(target[split:]), ancillae=target[:split])
    U += tq.gates.X(target=ancilla)

    for i in range(split, n):
        U += tq.gates.CNOT(control=ancilla, target=target[i])

    U += multi_controlled_not(target=ancilla, controls=target[:split], ancillae=target[split:])

    U += increment_circuit_n_ancillae(target=[ancilla] + list(target[split:]), ancillae=target[:split])
    U += tq.gates.X(target=ancilla)

    U += multi_controlled_not(target=ancilla, controls=target[:split], ancillae=target[split:])

    for i in range(split, n):
        U += tq.gates.CNOT(control=ancilla, target=target[i])

    U += increment_circuit_n_ancillae(target=list(target[:split]), ancillae=[ancilla] + list(target[split:]))

    return U


def increment_circuit_n_ancillae(target: Sequence[int], ancillae: Sequence[int]) -> tq.QCircuit:
    """
    Increments the target register.
    Reference: https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html

    :param target: Indices of the target qubits in LSB ordering.
    :param ancillae: Indices of the ancilla qubits. Must contain at least len(target) - 1 qubits.
    Ancillae can be in any state, and will be returned to that state by the end of the circuit.
    :return: A circuit implementing the increment.
    """
    assert len(ancillae) >= len(target) - 1
    assert len(set(target) & set(ancillae)) == 0

    # If there are more than n - 1 ancillas, ignore them
    ancillae = ancillae[: len(target) - 1]

    if len(target) == 1:
        return tq.gates.X(target=target[0])

    if len(target) == 2:
        return tq.gates.CNOT(target=target[1], control=target[0]) + tq.gates.X(target=target[0])

    U = tq.QCircuit()

    for i in range(0, len(ancillae)):
        U += tq.gates.X(target=ancillae[i])

    U += tq.gates.X(target=target[-1])

    # No need for a separate subtraction widget construction as in the
    # blog post, we can simply use the adjoint of our addition circuit.
    U += addition_circuit(target=target, source=ancillae).dagger()

    for i in range(len(ancillae)):
        U += tq.gates.X(target=ancillae[i])

    U += addition_circuit(target=target, source=ancillae).dagger()

    return U


def multi_controlled_not(
    target: int, controls: Sequence[int], ancillae: Sequence[int], uncompute: bool = True
) -> tq.QCircuit:
    """
    Implements a multi-controlled NOT gate using Toffoli gates.

    :param target: Index of the target qubit.
    :param controls: Indices of the control qubits.
    :param ancillae: Indices of the ancilla qubits.
    Must contain at least len(controls) - 2 qubits. Ancillae can be in any state,
    and will be returned to that state by the end of the circuit, except if uncompute is set to False.
    :param uncompute: Whether to uncompute the ancillae.
    Can be used an optimization if the circuit is used twice.
    :return: A circuit implementing the multi-controlled NOT gate.
    """
    assert len(ancillae) >= len(controls) - 2
    assert set(ancillae).isdisjoint(controls)
    assert target not in set(controls) | set(ancillae)

    if len(controls) <= 2:
        return tq.gates.X(target=target, control=controls)

    staircase = tq.QCircuit()
    for i in range(2, len(controls) - 1):
        staircase += tq.gates.Toffoli(first=controls[i], second=ancillae[i - 2], target=ancillae[i - 1])

    U = tq.QCircuit()

    U += tq.gates.Toffoli(first=controls[-1], second=ancillae[len(controls) - 3], target=target)
    U += staircase.dagger()
    U += tq.gates.Toffoli(first=controls[0], second=controls[1], target=ancillae[0])
    U += copy.deepcopy(staircase)
    U += tq.gates.Toffoli(first=controls[-1], second=ancillae[len(controls) - 3], target=target)

    if uncompute:
        U += staircase.dagger()
        U += tq.gates.Toffoli(first=controls[0], second=controls[1], target=ancillae[0])
        U += staircase

    return U


def _carry_circuit(target: Sequence[int], const: int, carry: int, ancillae: Sequence[int]) -> tq.QCircuit:
    """
    Implements the carry circuit from https://arxiv.org/abs/1611.07995, Fig. 3. Used for const_addition_circuit.
    Expects LSB ordering.
    """
    assert len(ancillae) >= len(target) - 1
    assert len(set(target) & set(ancillae)) == 0
    assert carry not in set(target) | set(ancillae)
    n = len(target)

    U = tq.QCircuit()

    if n == 1:
        if const & 1:
            U += tq.gates.CNOT(control=target[0], target=carry)
        return U

    U += tq.gates.CNOT(control=ancillae[n - 2], target=carry)

    half = tq.QCircuit()

    for i in reversed(range(1, n)):
        if const & (1 << i):
            half += tq.gates.CNOT(control=target[i], target=ancillae[i - 1])
            half += tq.gates.X(target=target[i])
        if i > 1:
            half += tq.gates.Toffoli(first=ancillae[i - 2], second=target[i], target=ancillae[i - 1])

    if const & 1:
        half += tq.gates.Toffoli(first=target[0], second=target[1], target=ancillae[0])

    for i in range(2, n):
        half += tq.gates.Toffoli(first=ancillae[i - 2], second=target[i], target=ancillae[i - 1])

    U += half

    U += tq.gates.CNOT(control=ancillae[n - 2], target=carry)

    U += half.dagger()

    return U


def const_addition_circuit(target: Sequence[int], const: int, ancillae: Sequence[int]) -> tq.QCircuit:
    """
    Adds a constant to the target register.
    Reference: https://arxiv.org/abs/1611.07995, chapter 2

    :param target: Indices of the target qubits in LSB ordering.
    :param const: The constant to be added.
    :param ancillae: Indices of the ancilla qubits. Must contain at least 2 qubits.
    Ancillae can be in any state, and will be returned to that state by the end of the circuit.
    :return: A circuit implementing the addition.
    """
    n = len(target)
    if n == 1:
        U = tq.QCircuit()
        if const & 1:
            U += tq.gates.X(target=target[0])
        return U

    assert abs(const) < 2**n
    assert len(ancillae) >= 2

    split = (n + 1) // 2

    U = tq.QCircuit()

    U += increment_circuit_n_ancillae(
        target=[ancillae[0]] + list(target[split:]), ancillae=list(target[:split]) + [ancillae[1]]
    )
    U += tq.gates.X(target=ancillae[0])

    for i in range(split, n):
        U += tq.gates.CNOT(control=ancillae[0], target=target[i])

    U += _carry_circuit(
        target=target[:split], const=const % (2**split), carry=ancillae[0], ancillae=list(target[split:])
    )

    U += increment_circuit_n_ancillae(
        target=[ancillae[0]] + list(target[split:]), ancillae=list(target[:split]) + [ancillae[1]]
    )
    U += tq.gates.X(target=ancillae[0])

    U += _carry_circuit(
        target=target[:split], const=const % (2**split), carry=ancillae[0], ancillae=list(target[split:])
    )

    for i in range(split, n):
        U += tq.gates.CNOT(control=ancillae[0], target=target[i])

    U += const_addition_circuit(target=target[split:], const=const >> split, ancillae=ancillae)
    U += const_addition_circuit(target=target[:split], const=const % (2**split), ancillae=ancillae)

    return U
