import pytest

from bequem.qubit_map import QubitMap, Qubit, ID, ZeroQubit
from bequem.nodes.permutation import _find_matching_partitioning, Permutation, PermuteRegisters


def test_find_permutation_trivial():
    Permutation(QubitMap(0), QubitMap(0)).verify()
    Permutation(QubitMap(0, 1), QubitMap(0, 1)).verify()
    Permutation(QubitMap(1), QubitMap(1)).verify()
    Permutation(QubitMap(1, 1), QubitMap(1, 1)).verify()
    Permutation(QubitMap(1, 2), QubitMap(1, 2)).verify()
    c = Qubit(QubitMap(1), QubitMap(0, 1))
    Permutation(QubitMap([c]), QubitMap([c])).verify()

    Permutation(QubitMap(0), QubitMap(0, 1)).verify()
    Permutation(QubitMap(1), QubitMap(1, 1)).verify()
    Permutation(QubitMap([c]), QubitMap([c], 1)).verify()


def test_find_permutation_matching_partitioning():
    Permutation(QubitMap([ZeroQubit(), ID]), QubitMap(1)).verify()
    Permutation(QubitMap(1), QubitMap([ZeroQubit(), ID])).verify()


@pytest.mark.xfail
def test_brute_force_1_simple_rotation():
    a = QubitMap(1)
    a1 = QubitMap(1, 1)
    b = QubitMap([Qubit(a, a)])
    c = QubitMap([Qubit(b, a1)])
    d = QubitMap([Qubit(a1, b)])
    Permutation(d, c).verify()
    Permutation(c, d).verify()

@pytest.mark.xfail
def test_brute_force_2_simple_rotations():
    a = QubitMap(2)

    # Left
    b1 = QubitMap(1)
    b2 = QubitMap([Qubit(b1, QubitMap(0, 1))])
    b3 = QubitMap([Qubit(b2, QubitMap(0, 2))])
    Permutation(a, b3).verify()
    Permutation(b3, a).verify()

    # Right
    b1 = QubitMap(1)
    b2 = QubitMap([Qubit(QubitMap(0, 1), b1)])
    b3 = QubitMap([Qubit(QubitMap(0, 2), b2)])
    Permutation(a, b3).verify()
    Permutation(b3, a).verify()

@pytest.mark.xfail
def test_brute_force_double_rotation_left_right():
    a = QubitMap(2)

    b1 = QubitMap(1)
    b2 = QubitMap([Qubit(QubitMap(0, 1), b1)])
    b3 = QubitMap([Qubit(b2, QubitMap(0, 2))])
    Permutation(a, b3).verify()

@pytest.mark.xfail
def test_brute_force_double_rotation_right_left():
    a = QubitMap(2)

    b1 = QubitMap(1)
    b2 = QubitMap([Qubit(b1, QubitMap(0, 1))])
    b3 = QubitMap([Qubit(QubitMap(0, 2), b2)])
    Permutation(a, b3).verify()


def test_permute_registers():
    PermuteRegisters(QubitMap(1), [0]).verify()
    PermuteRegisters(QubitMap(2), [0, 1]).verify()
    PermuteRegisters(QubitMap(2), [1, 0]).verify()
    PermuteRegisters(QubitMap(3), [0, 1, 2]).verify()
    PermuteRegisters(QubitMap(3), [0, 2, 1]).verify()
    PermuteRegisters(QubitMap(3), [1, 0, 2]).verify()
    PermuteRegisters(QubitMap(3), [1, 2, 0]).verify()
    PermuteRegisters(QubitMap(3), [2, 0, 1]).verify()
    PermuteRegisters(QubitMap(3), [2, 1, 0]).verify()


def test_find_matching_partitioning():
    assert _find_matching_partitioning(QubitMap(0), QubitMap(0)) == []
    assert _find_matching_partitioning(QubitMap(1), QubitMap(1)) == [
        (QubitMap(1), QubitMap(1))
    ]
    assert _find_matching_partitioning(QubitMap(2), QubitMap(2)) == [
        (QubitMap(1), QubitMap(1)),
        (QubitMap(1), QubitMap(1)),
    ]

    c = Qubit(QubitMap(1), QubitMap(1))
    assert _find_matching_partitioning(QubitMap([c]), QubitMap(2)) == [
        (QubitMap(1), QubitMap(1)),
        (QubitMap(1), QubitMap(1)),
    ]
    assert _find_matching_partitioning(QubitMap(2), QubitMap([c])) == [
        (QubitMap(1), QubitMap(1)),
        (QubitMap(1), QubitMap(1)),
    ]
    c = Qubit(QubitMap(1), QubitMap(0, 1))
    assert _find_matching_partitioning(QubitMap([c]), QubitMap([c])) == [
        (QubitMap([c]), QubitMap([c]))
    ]
    assert _find_matching_partitioning(QubitMap([ID, c]), QubitMap([c, ID])) == [
        (QubitMap([ID, c]), QubitMap([c, ID]))
    ]
    assert _find_matching_partitioning(QubitMap([ID, c]), QubitMap([ID, c])) == [
        (QubitMap(1), QubitMap(1)),
        (QubitMap([c]), QubitMap([c])),
    ]
