import pytest

import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Qubit, ID
from bequem.nodes.permutation import _find_matching_subdivision, find_permutation


def check_find_permutation(a: QubitMap, b: QubitMap):
    print(f"permuting\n{a}\nand\n{b}")
    permutation = find_permutation(a, b)
    assert permutation.permute_a.qubits_in().registers == a.registers
    assert permutation.permute_b.qubits_in().registers == b.registers
    permutation.verify()


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


@pytest.mark.xfail
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
        (QubitMap(1), QubitMap(1)),
        (QubitMap(1), QubitMap(1)),
    ]
    assert _find_matching_subdivision(QubitMap(2), QubitMap([c])) == [
        (QubitMap(1), QubitMap(1)),
        (QubitMap(1), QubitMap(1)),
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
