import tequila as tq

from .qubit_map import QubitMap, Qubit
from .circuit import Circuit


def find_permutation(a: QubitMap, b: QubitMap) -> tuple[Circuit, Circuit, QubitMap]:
    if a.dimension != b.dimension:
        raise ValueError()

    for (sub_a, sub_b) in _find_matching_subdivision(a, b):
        if sub_a == sub_b:
            continue

    return Circuit(), Circuit(), a


def _find_matching_subdivision(
    a: QubitMap, b: QubitMap
) -> list[tuple[QubitMap, QubitMap]]:
    if a.registers == [] and b.registers == []:
        return []

    assert a.dimension == b.dimension
    subdivisions = []
    last_breakpoint_a = 0
    last_breakpoint_b = 0
    i_a = 1
    i_b = 1
    submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
    submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])
    while i_a < len(a.registers) and i_b < len(b.registers):
        if submap_a.dimension == submap_b.dimension:
            subdivisions.append((submap_a, submap_b))
            last_breakpoint_a = i_a
            last_breakpoint_b = i_b
            i_a += 1
            i_b += 1
            submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
            submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])
        elif submap_a.dimension < submap_b.dimension:
            i_a += 1
            submap_a = QubitMap(a.registers[last_breakpoint_a:i_a])
        else:
            i_b += 1
            submap_b = QubitMap(b.registers[last_breakpoint_b:i_b])

    submap_a = QubitMap(a.registers[last_breakpoint_a:])
    submap_b = QubitMap(b.registers[last_breakpoint_b:])
    subdivisions.append((submap_a, submap_b))

    return subdivisions


def _desimplify(qubits: QubitMap) -> QubitMap:
    assert len(qubits.registers) >= 2
    tail = qubits.registers[-1]
    assert isinstance(tail, Qubit)
    tail2 = qubits.registers[-2]

    c = Qubit(
        [tail2] + tail.case_zero.registers,
        [tail2] + tail.case_one.registers,
    )
    return QubitMap(qubits.registers[:-2] + [c])


def _find_permutation_brute_force(
    a: QubitMap, b: QubitMap
) -> tuple[Circuit, Circuit, QubitMap]:
    assert a.dimension == b.dimension
    assert len(a.registers) != 0
    assert len(b.registers) != 0
    tail_a = a.registers[-1]
    assert isinstance(tail_a, Qubit)
    tail_b = b.registers[-1]
    assert isinstance(tail_b, Qubit)

    # TODO: Potentially switch a and b

    head_a = QubitMap(a.registers[:-1])
    head_a_dim = head_a.dimension
    head_b = QubitMap(a.registers[:-1])
    head_b_dim = head_b.dimension
    while head_a_dim % head_b_dim != 0:
        b = _desimplify(b)
        tail_b = b.registers[-1]
        head_b = QubitMap(a.registers[:-1])
        head_b_dim = head_b.dimension

    while tail_a.case_zero.dimension * head_a_dim != tail_b.case_zero.dimension * head_b_dim:
        if tail_a.case_zero.dimension * head_a_dim < tail_b.case_zero.dimension * head_b_dim:
            assert len(tail_b.case_zero.registers) > 0
            pivot = tail_b.case_zero.registers[-1]
            assert isinstance(pivot, Qubit)
            if tail_a.case_zero.dimension * head_a_dim <= pivot.case_zero.dimension * head_b_dim:
                # Right rotation
                raise NotImplementedError
            else:
                # Left-right rotation
                assert len(pivot.case_one.registers) > 0
                pivot2 = pivot.case_one.registers[-1]
                assert isinstance(pivot2, Qubit)
                raise NotImplementedError
        else:
            assert len(tail_b.case_one.registers) > 0
            pivot = tail_b.case_one.registers[-1]
            assert isinstance(pivot, Qubit)
            if tail_a.case_one.dimension * head_a_dim <= pivot.case_one.dimension * head_b_dim:
                # Left rotation
                raise NotImplementedError
            else:
                # Right-left rotation
                assert len(pivot.case_zero.registers) > 0
                pivot2 = pivot.case_zero.registers[-1]
                assert isinstance(pivot2, Qubit)
                raise NotImplementedError

    control_a = a.total_qubits - 1
    control_b = b.total_qubits - 1

    perm_zero_a, perm_zero_b, map_zero = find_permutation(
        QubitMap(head_a.registers + tail_a.case_zero.registers),
        QubitMap(head_b.registers + tail_b.case_zero.registers),
    )
    perm_zero_a.tq_circuit.add_controls(control_a)
    perm_zero_b.tq_circuit.add_controls(control_b)

    if tail_a.case_zero == tail_a.case_one and tail_b.case_zero == tail_b.case_one:
        perm_one_a = perm_zero_a
        perm_one_b = perm_one_a
        map_one = map_zero
    else:
        perm_one_a, perm_one_b, map_one = find_permutation(
            QubitMap(head_a.registers + tail_a.case_one.registers),
            QubitMap(head_b.registers + tail_b.case_one.registers),
        )
        perm_one_a.tq_circuit.add_controls(control_a)
        perm_one_b.tq_circuit.add_controls(control_b)

    perm_a = Circuit()
    perm_a.tq_circuit += tq.gates.X(control_a)
    perm_a.tq_circuit += perm_zero_a
    perm_a.tq_circuit += tq.gates.X(control_a)
    perm_a.tq_circuit += perm_one_a

    perm_b = Circuit()
    perm_b.tq_circuit += tq.gates.X(control_b)
    perm_b.tq_circuit += perm_zero_b
    perm_b.tq_circuit += tq.gates.X(control_b)
    perm_b.tq_circuit += perm_one_b

    return perm_a, perm_b, Qubit(map_zero, map_one).simplify()
