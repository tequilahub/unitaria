import pytest

from bequem.qubit_map import QubitMap, Qubit, ID, ZeroQubit
from bequem.nodes.permutation import _find_matching_subdivision, Permutation


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


@pytest.mark.xfail
def test_find_permutation_matching_subdivision():
    Permutation(QubitMap([ZeroQubit(), ID]), QubitMap(1)).verify()


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
