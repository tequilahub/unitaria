import tequila as tq

from .qubit_map import QubitMap, Qubit, ID
from .circuit import Circuit


def find_permutation(a: QubitMap,
                     b: QubitMap) -> tuple[Circuit, Circuit, QubitMap]:
    if a.dimension != b.dimension:
        raise ValueError()

    perm_a = Circuit()
    perm_b = Circuit()

    registers = []
    zero_bits = max(a.zero_qubits, b.zero_qubits)
    total_bits = 0

    for (sub_a, sub_b) in _find_matching_subdivision(a, b):
        if sub_a == sub_b:
            registers += sub_a.registers
            total_bits += sub_a.total_qubits
            continue
        circ_a, circ_b, new_sub = _find_permutation_brute_force(sub_a, sub_b)
        circ_a.tq_circuit.map_qubits(
            dict([(i, i + total_bits) for i in circ_a.tq_circuit.qubits]))
        circ_b.tq_circuit.map_qubits(
            dict([(i, i + total_bits) for i in circ_b.tq_circuit.qubits]))
        perm_a += circ_a
        perm_b += circ_b
        registers += new_sub.registers
        # TODO: Permute zero bits
        zero_bits += new_sub.zero_qubits
        total_bits += new_sub.total_qubits

    return perm_a, perm_b, QubitMap(
        registers, zero_bits + max(a.zero_qubits, b.zero_qubits))


def _find_matching_subdivision(a: QubitMap,
                               b: QubitMap) -> list[tuple[QubitMap, QubitMap]]:
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


def _simplify(qubits: QubitMap) -> tuple[QubitMap, Circuit]:
    if len(qubits.registers) > 0 and isinstance(qubits.registers[-1], Qubit):
        # case_zero, circ_zero = _simplify(qubits.registers[-1].case_zero)
        # case_one, circ_one = _simplify(qubits.registers[-1].case_one)
        case_zero = qubits.registers[-1].case_zero
        case_one = qubits.registers[-1].case_one

        common_zeros = min(case_zero.zero_qubits, case_one.zero_qubits)
        common_tail = []
        for i in range(min(len(case_zero.registers), len(case_one.registers))):
            if case_zero.registers[i] == case_one.registers[i]:
                common_tail.append(case_zero.registers[i])
        case_zero = QubitMap(case_zero.registers[len(common_tail):],
                             case_zero.zero_qubits - common_zeros)
        case_one = QubitMap(case_one.registers[len(common_tail):],
                            case_one.zero_qubits - common_zeros)
        circuit = Circuit()
        if common_zeros > 0:
            circuit += Circuit(
                tq.gates.SWAP(
                    qubits.total_qubits - qubits.zero_qubits - 1,
                    qubits.total_qubits - qubits.zero_qubits - common_zeros -
                    1))
        return QubitMap(
            qubits.registers[:-1] + common_tail + [Qubit(case_zero, case_one)],
            qubits.zero_qubits + common_zeros), circuit

    return qubits, Circuit()


def _find_permutation_brute_force(
        a: QubitMap, b: QubitMap) -> tuple[Circuit, Circuit, QubitMap]:
    assert a.dimension == b.dimension
    assert len(a.registers) != 0
    assert len(b.registers) != 0

    # TODO: Potentially switch a and b
    perm_b = Circuit()

    while _split_dimension(a) != _split_dimension(b):
        if _split_dimension(a) < _split_dimension(b):
            pivot = _left_child(b)
            if _split_dimension(a) <= _split_dimension(pivot):
                # Right rotation
                left = _extend_zeros(_left_child(pivot), 1)
                middle = _right_child(pivot)
                right = _right_child(b)
                if right.zero_qubits == 0:
                    left = _extend_zeros(left, 1)
                    middle = _extend_zeros(middle, 1)
                    last_bit = left.total_qubits
                    perm_b.tq_circuit += tq.gates.SWAP(last_bit - 1, last_bit)
                    perm_b.tq_circuit += tq.gates.SWAP(last_bit - 2,
                                                       last_bit - 1)
                else:
                    right = _extend_zeros(right, -1)
                last_bit = left.total_qubits
                perm_b.tq_circuit += tq.gates.CNOT(last_bit - 1, last_bit)
                perm_b.tq_circuit += tq.gates.CNOT(last_bit, last_bit - 1)

                b = QubitMap([Qubit(left, QubitMap([Qubit(middle, right)]))])

                b, simp_b = _simplify(b)
                perm_b += simp_b
            else:
                # Left-right rotation
                pivot2 = _right_child(pivot)
                raise NotImplementedError
        else:
            pivot = _right_child(b)
            if _split_dimension(a) >= _split_dimension(pivot):
                # Left rotation
                right = _left_child(b)
                middle = _left_child(pivot)
                left = _extend_zeros(_right_child(pivot), 1)
                if left.zero_qubits == 0:
                    right = _extend_zeros(right, 1)
                    middle = _extend_zeros(middle, 1)
                    last_bit = right.total_qubits
                    perm_b.tq_circuit += tq.gates.SWAP(last_bit - 1, last_bit)
                    perm_b.tq_circuit += tq.gates.SWAP(last_bit - 2,
                                                       last_bit - 1)
                else:
                    left = _extend_zeros(left, -1)
                last_bit = right.total_qubits
                perm_b.tq_circuit += tq.gates.CNOT(last_bit, last_bit - 1)
                perm_b.tq_circuit += tq.gates.CNOT(last_bit - 1, last_bit)

                b = QubitMap([Qubit(QubitMap([Qubit(left, middle)]), right)])

                b, simp_b = _simplify(b)
                perm_b += simp_b
            else:
                # Right-left rotation
                pivot2 = _left_child(pivot)
                raise NotImplementedError

    control_a = a.total_qubits - a.zero_qubits - 1
    control_b = b.total_qubits - b.zero_qubits - 1

    a_zero = _left_child(a)
    b_zero = _left_child(b)
    a_one = _right_child(a)
    b_one = _right_child(b)

    perm_zero_a, perm_zero_b, map_zero = find_permutation(a_zero, b_zero)

    if a_zero == a_one and b_zero == b_one:
        return perm_zero_a, perm_zero_b, QubitMap(map_zero.registers + [ID],
                                                  map_zero.zero_qubits)

    perm_one_a, perm_one_b, map_one = find_permutation(a_one, b_one)

    perm_zero_a.tq_circuit.add_controls(control_a)
    perm_zero_b.tq_circuit.add_controls(control_b)
    perm_one_a.tq_circuit.add_controls(control_a)
    perm_one_b.tq_circuit.add_controls(control_b)

    perm_a = Circuit()
    perm_a.tq_circuit += tq.gates.X(control_a)
    perm_a.tq_circuit += perm_zero_a.tq_circuit
    perm_a.tq_circuit += tq.gates.X(control_a)
    perm_a.tq_circuit += perm_one_a.tq_circuit

    perm_b.tq_circuit += tq.gates.X(control_b)
    perm_b.tq_circuit += perm_zero_b.tq_circuit
    perm_b.tq_circuit += tq.gates.X(control_b)
    perm_b.tq_circuit += perm_one_b.tq_circuit

    return perm_a, perm_b, QubitMap([Qubit(map_zero, map_one)])


def _split_dimension(qubits: QubitMap) -> int:
    assert len(qubits.registers) != 0
    assert isinstance(qubits.registers[-1], Qubit)

    total_dim = qubits.dimension
    zero_dim = qubits.registers[-1].case_zero.dimension
    one_dim = qubits.registers[-1].case_one.dimension

    return total_dim / (zero_dim + one_dim) * zero_dim


def _left_child(qubits: QubitMap) -> QubitMap:
    assert len(qubits.registers) != 0
    assert isinstance(qubits.registers[-1], Qubit)

    return QubitMap(
        qubits.registers[:-1] + qubits.registers[-1].case_zero.registers,
        qubits.zero_qubits + qubits.registers[-1].case_zero.zero_qubits)


def _right_child(qubits: QubitMap) -> QubitMap:
    assert len(qubits.registers) != 0
    assert isinstance(qubits.registers[-1], Qubit)

    return QubitMap(
        qubits.registers[:-1] + qubits.registers[-1].case_one.registers,
        qubits.zero_qubits + qubits.registers[-1].case_one.zero_qubits)


def _extend_zeros(qubits: QubitMap, n: int) -> QubitMap:
    return QubitMap(qubits.registers[:], qubits.zero_qubits + n)
