import copy
from typing import Sequence

import tequila as tq


def addition_circuit(source: Sequence[int], target: Sequence[int]) -> tq.QCircuit:
    """
    Adds the source register to the target register.
    Reference: https://arxiv.org/abs/0910.2530v1

    :param source: Indices of the source qubits in MSB ordering.
    Requires len(source) >= 2.
    :param target: Indices of the target qubits in MSB ordering.
    Requires len(target) >= len(source).
    :return: A circuit implementing the addition.
    """
    assert 2 <= len(source) <= len(target)
    assert set(source).isdisjoint(set(target))
    n = len(source)

    # Change register to LSB ordering and name them like in the paper.
    # Note that unlike in the paper, A_n does not exist, because of the special handling for larger target registers.
    a = source[::-1]
    b = target[::-1]

    U = tq.QCircuit()

    for i in range(1, n):
        U += tq.gates.CNOT(control=a[i], target=b[i])

    if len(target) > n:
        U += increment_circuit_single_ancilla(target=list(b[n:][::-1]) + [a[n - 1]], ancilla=a[0], second_ancilla=b[0])
        U += tq.gates.X(target=a[n - 1])
        for i in range(n, len(b)):
            U += tq.gates.CNOT(control=a[n - 1], target=b[i])

    for i in reversed(range(1, n - 1)):
        U += tq.gates.CNOT(control=a[i], target=a[i + 1])
    for i in range(n - 1):
        U += tq.gates.Toffoli(first=a[i], second=b[i], target=a[i + 1])

    if len(target) > n:
        U += increment_circuit_single_ancilla(target=list(b[n:][::-1]) + [a[n - 1], b[n - 1]], ancilla=a[0], second_ancilla=b[0])
        U += tq.gates.X(target=b[n - 1])
        U += tq.gates.CNOT(control=b[n - 1], target=a[n - 1])

    for i in reversed(range(1, n)):
        U += tq.gates.CNOT(control=a[i], target=b[i])
        U += tq.gates.Toffoli(first=a[i - 1], second=b[i - 1], target=a[i])
    for i in range(1, n - 1):
        U += tq.gates.CNOT(control=a[i], target=a[i + 1])
    for i in range(n):
        U += tq.gates.CNOT(control=a[i], target=b[i])

    if len(target) > n:
        for i in range(n, len(b)):
            U += tq.gates.CNOT(control=a[n - 1], target=b[i])

    return U


def increment_circuit_single_ancilla(target: Sequence[int], ancilla: int, second_ancilla: int = None) -> tq.QCircuit:
    """
    Increments the target register.
    Reference: https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html

    :param target: Indices of the target qubits in MSB ordering.
    :param ancilla: Index of the ancilla qubit.
    Can be in any state, and will be returned to that state by the end of the circuit.
    :param second_ancilla: Optional second ancilla qubit.
    Is not required, but can be used for a more efficient construction when len(target) is even.
    :return: A circuit implementing the increment.

    The ancilla qubits can be in any state, and will be returned to that state by the end of the circuit.
    Expects MSB ordering.
    """
    assert ancilla not in target
    if second_ancilla is not None:
        assert second_ancilla != ancilla and second_ancilla not in target

    split = len(target) // 2

    U = tq.QCircuit()

    upper_inc = tq.QCircuit()
    if len(target) % 2 == 0:
        if second_ancilla is not None:
            upper_inc += increment_circuit_n_ancillae(target=list(target[:split]) + [ancilla],
                                                      ancillae=list(target[split:]) + [second_ancilla])
        else:
            upper_inc += multi_controlled_not(target=target[0], controls=list(target[1:split]) + [ancilla], ancillae=target[split:])
            upper_inc += increment_circuit_n_ancillae(target=list(target[1:split]) + [ancilla], ancillae=list(target[split:]))
    else:
        upper_inc += increment_circuit_n_ancillae(target=list(target[:split]) + [ancilla], ancillae=target[split:])

    U += copy.deepcopy(upper_inc)
    U += tq.gates.X(target=ancilla)

    for i in range(split):
        U += tq.gates.CNOT(control=ancilla, target=target[i])

    U += multi_controlled_not(target=ancilla, controls=target[split:], ancillae=target[:split])

    U += upper_inc
    U += tq.gates.X(target=ancilla)

    U += multi_controlled_not(target=ancilla, controls=target[split:], ancillae=target[:split])

    for i in range(split):
        U += tq.gates.CNOT(control=ancilla, target=target[i])

    U += increment_circuit_n_ancillae(target=list(target[split:]), ancillae=list(target[:split]) + [ancilla])

    return U


def increment_circuit_n_ancillae(target: Sequence[int], ancillae: Sequence[int]) -> tq.QCircuit:
    """
    Increments the target register.
    Reference: https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html

    :param target: Indices of the target qubits in MSB ordering.
    :param ancillae: Indices of the ancilla qubits. Must contain at least len(target) qubits.
    Ancillae can be in any state, and will be returned to that state by the end of the circuit.
    :return: A circuit implementing the increment.
    """
    assert len(target) <= len(ancillae)
    assert len(set(target) & set(ancillae)) == 0

    v = target[::-1]  # LSB ordering

    # If there are more than n ancillas, ignore them
    g = ancillae[:len(target)]

    U = tq.QCircuit()

    for i in range(len(v)):
        U += tq.gates.CNOT(control=g[0], target=v[i])

    for i in range(1, len(g)):
        U += tq.gates.X(target=g[i])

    U += tq.gates.X(target=v[-1])

    U += _subtraction_widget(v, g[1:], g[0])

    for i in range(1, len(g)):
        U += tq.gates.X(target=g[i])

    U += _subtraction_widget(v, g[1:], g[0])

    for i in range(len(v)):
        U += tq.gates.CNOT(control=g[0], target=v[i])

    return U


def _subtraction_widget(v: Sequence[int], g: Sequence[int], c: int) -> tq.QCircuit:
    """
    Implements the subtraction widget from https://algassert.com/circuits/2015/06/12/Constructing-Large-Increment-Gates.html.
    Stores v - g - c in v and leaves the other registers unchanged. Used for increment_circuit_n_ancillas.
    Expects LSB ordering.
    """
    g = [c] + list(g)

    assert len(v) == len(g)
    assert len(set(v) & set(g)) == 0

    U = tq.QCircuit()

    for i in range(len(v) - 1):
        U += tq.gates.CNOT(control=g[i], target=v[i])
        U += tq.gates.CNOT(control=g[i + 1], target=g[i])
        U += tq.gates.Toffoli(first=g[i], second=v[i], target=g[i + 1])

    U += tq.gates.CNOT(control=g[-1], target=v[-1])

    for i in reversed(range(len(v) - 1)):
        U += tq.gates.Toffoli(first=g[i], second=v[i], target=g[i + 1])
        U += tq.gates.CNOT(control=g[i + 1], target=g[i])
        U += tq.gates.CNOT(control=g[i + 1], target=v[i])

    return U


def multi_controlled_not(target: int, controls: Sequence[int], ancillae: Sequence[int],
                         uncompute: bool = True) -> tq.QCircuit:
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
    U += staircase
    U += tq.gates.Toffoli(first=controls[-1], second=ancillae[len(controls) - 3], target=target)

    if uncompute:
        U += staircase.dagger()
        U += tq.gates.Toffoli(first=controls[0], second=controls[1], target=ancillae[0])
        U += staircase

    return U
