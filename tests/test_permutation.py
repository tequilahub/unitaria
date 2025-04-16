import numpy as np

from bequem.qubit_map import QubitMap, Qubit, ID
from bequem.permutation import _find_matching_subdivision, find_permutation


def check_find_permutation(a: QubitMap, b: QubitMap):
    perm_a, perm_b, result = find_permutation(a, b)
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


def test_find_permutation():
    check_find_permutation(QubitMap(0), QubitMap(0))
    check_find_permutation(QubitMap(0, 1), QubitMap(0, 1))
    check_find_permutation(QubitMap(1), QubitMap(1))
    check_find_permutation(QubitMap(1, 1), QubitMap(1, 1))
    check_find_permutation(QubitMap(1, 2), QubitMap(1, 2))
    c = Qubit(QubitMap(1), QubitMap(0, 1))
    check_find_permutation(QubitMap([c]), QubitMap([c]))


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
