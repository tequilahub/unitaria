import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Qubit, ID
from bequem.permutation import _find_matching_subdivision, find_permutation


def check_find_permutation(a: QubitMap, b: QubitMap):
    print(f"permuting\n{a}\nand\n{b}")
    perm_a, perm_b, result = find_permutation(a, b)
    for i in range(result.total_qubits):
        perm_a.tq_circuit += tq.gates.Phase(i, angle=0)
        perm_b.tq_circuit += tq.gates.Phase(i, angle=0)
    print("result:")
    print(result)
    tq.draw(perm_a.tq_circuit)
    tq.draw(perm_b.tq_circuit)
    assert result.dimension == a.dimension

    max_qubits = max(a.total_qubits, result.total_qubits)
    assert len(perm_a.tq_circuit.qubits) <= max_qubits

    for i in a.enumerate_basis():
        final_state = perm_a.simulate(i)
        i_permuted = None
        for j in range(2 ** result.total_qubits):
            if final_state[j] > 0.5:
                assert np.isclose(final_state[j], 1)
                i_permuted = j
                break
        assert i_permuted is not None, final_state
        assert result.test_basis(i_permuted)

    for i in b.enumerate_basis():
        final_state = perm_b.simulate(i)
        i_permuted = None
        for j in range(2 ** result.total_qubits):
            if final_state[j] > 0.5:
                assert np.isclose(final_state[j], 1)
                i_permuted = j
                break
        assert i_permuted is not None
        assert result.test_basis(i_permuted)


def test_find_permutation_matching_subdivision():
    check_find_permutation(QubitMap(0), QubitMap(0))
    check_find_permutation(QubitMap(0, 1), QubitMap(0, 1))
    check_find_permutation(QubitMap(1), QubitMap(1))
    check_find_permutation(QubitMap(1, 1), QubitMap(1, 1))
    check_find_permutation(QubitMap(1, 2), QubitMap(1, 2))
    c = Qubit(QubitMap(1), QubitMap(0, 1))
    check_find_permutation(QubitMap([c]), QubitMap([c]))

    check_find_permutation(QubitMap(0), QubitMap(0, 1))
    check_find_permutation(QubitMap(1), QubitMap(1, 1))
    check_find_permutation(QubitMap([c]), QubitMap([c], 1))


def test_find_permutation_brute_force():
    a = QubitMap(1)
    a1 = QubitMap(1, 1)
    b = QubitMap([Qubit(a, a)])
    c = QubitMap([Qubit(b, a1)])
    d = QubitMap([Qubit(a1, b)])
    check_find_permutation(d, c)
    check_find_permutation(c, d)


def test_find_matching_subdivision():
    assert _find_matching_subdivision(QubitMap(0), QubitMap(0)) == []
    assert _find_matching_subdivision(QubitMap(1), QubitMap(1)) == [
        (QubitMap(1), QubitMap(1))
    ]
    assert _find_matching_subdivision(QubitMap(2), QubitMap(2)) == [
        (QubitMap(1), QubitMap(1)),
        (QubitMap(1), QubitMap(1)),
    ]

    c = Qubit(QubitMap(1), QubitMap(1))
    assert _find_matching_subdivision(QubitMap([c]), QubitMap(2)) == [
        (QubitMap([c]), QubitMap(2))
    ]
    assert _find_matching_subdivision(QubitMap(2), QubitMap([c])) == [
        (QubitMap(2), QubitMap([c]))
    ]
    c = Qubit(QubitMap(1), QubitMap(0, 1))
    assert _find_matching_subdivision(QubitMap([c]), QubitMap([c])) == [
        (QubitMap([c]), QubitMap([c]))
    ]
    assert _find_matching_subdivision(QubitMap([ID, c]), QubitMap([c, ID])) == [
        (QubitMap([ID, c]), QubitMap([c, ID]))
    ]
    assert _find_matching_subdivision(QubitMap([ID, c]), QubitMap([ID, c])) == [
        (QubitMap(1), QubitMap(1)),
        (QubitMap([c]), QubitMap([c])),
    ]
