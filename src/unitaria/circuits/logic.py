from typing import Sequence

import tequila as tq


def multi_controlled_not(
    target: int,
    controls: Sequence[int],
    clean_ancillae: Sequence[int] = [],
    borrowed_ancillae: Sequence[int] = [],
    uncompute: bool = True,
) -> tq.QCircuit:
    """
    Implements a multi-controlled NOT gate using Toffoli gates.

    Will select the best variant depending on the ancillae available, see `multi_controlled_not_v_chain`, `multi_controlled_not_v_chain_borrowed`.

    :param target: Index of the target qubit.
    :param controls: Indices of the control qubits.
    :param clean_ancillae: Indices of the clean ancilla qubits. Clean ancillae
    should start in the 0 state, and will be returned to that state by the end
    of the circuit, except if uncompute is set to False.
    :param borrowed_ancillae: Indices of the borrowed ancilla qubits. Clean ancillae
    may start in any state, and will be returned to that state by the end of the
    circuit, except if uncompute is set to False.
    :param uncompute: Whether to uncompute the ancillae.
    Can be used an optimization if the circuit is used twice.
    :return: A circuit implementing the multi-controlled NOT gate.
    """
    if len(clean_ancillae) >= len(controls) - 2:
        return multi_controlled_not_v_chain(target, controls, clean_ancillae, uncompute)
    if len(clean_ancillae) + len(borrowed_ancillae) >= len(controls) - 2:
        return multi_controlled_not_v_chain_borrowed(target, controls, clean_ancillae + borrowed_ancillae, uncompute)
    return tq.gates.X(target=target, control=controls)


def multi_controlled_not_v_chain(
    target: int, controls: Sequence[int], ancillae: Sequence[int], uncompute: bool = True
) -> tq.QCircuit:
    """
    Implements a multi-controlled NOT gate using Toffoli gates.

    :param target: Index of the target qubit.
    :param controls: Indices of the control qubits.
    :param ancillae: Indices of the ancilla qubits.
    Must contain at least len(controls) - 2 qubits. Ancillae should start in the 0 state,
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

    U += tq.gates.Toffoli(first=controls[0], second=controls[1], target=ancillae[0])
    U += staircase
    U += tq.gates.Toffoli(first=controls[-1], second=ancillae[len(controls) - 3], target=target)

    if uncompute:
        U += staircase.dagger()
        U += tq.gates.Toffoli(first=controls[0], second=controls[1], target=ancillae[0])

    return U


def multi_controlled_not_v_chain_borrowed(
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
    U += staircase
    U += tq.gates.Toffoli(first=controls[-1], second=ancillae[len(controls) - 3], target=target)

    if uncompute:
        U += staircase.dagger()
        U += tq.gates.Toffoli(first=controls[0], second=controls[1], target=ancillae[0])
        U += staircase

    return U
