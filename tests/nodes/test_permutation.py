import pytest

from bequem.qubit_map import Subspace, ControlledSubspace, ID, ZeroQubit
from bequem.nodes.permutation import _find_matching_partitioning, Permutation, PermuteRegisters


def test_find_permutation_trivial():
    Permutation(Subspace(0), Subspace(0)).verify()
    Permutation(Subspace(0, 1), Subspace(0, 1)).verify()
    Permutation(Subspace(1), Subspace(1)).verify()
    Permutation(Subspace(1, 1), Subspace(1, 1)).verify()
    Permutation(Subspace(1, 2), Subspace(1, 2)).verify()
    c = ControlledSubspace(Subspace(1), Subspace(0, 1))
    Permutation(Subspace([c]), Subspace([c])).verify()

    Permutation(Subspace(0), Subspace(0, 1)).verify()
    Permutation(Subspace(1), Subspace(1, 1)).verify()
    Permutation(Subspace([c]), Subspace([c], 1)).verify()


def test_find_permutation_matching_partitioning():
    Permutation(Subspace([ZeroQubit(), ID]), Subspace(1)).verify()
    Permutation(Subspace(1), Subspace([ZeroQubit(), ID])).verify()


@pytest.mark.xfail
def test_brute_force_1_simple_rotation():
    a = Subspace(1)
    a1 = Subspace(1, 1)
    b = Subspace([ControlledSubspace(a, a)])
    c = Subspace([ControlledSubspace(b, a1)])
    d = Subspace([ControlledSubspace(a1, b)])
    Permutation(d, c).verify()
    Permutation(c, d).verify()

@pytest.mark.xfail
def test_brute_force_2_simple_rotations():
    a = Subspace(2)

    # Left
    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(b1, Subspace(0, 1))])
    b3 = Subspace([ControlledSubspace(b2, Subspace(0, 2))])
    Permutation(a, b3).verify()
    Permutation(b3, a).verify()

    # Right
    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(Subspace(0, 1), b1)])
    b3 = Subspace([ControlledSubspace(Subspace(0, 2), b2)])
    Permutation(a, b3).verify()
    Permutation(b3, a).verify()

@pytest.mark.xfail
def test_brute_force_double_rotation_left_right():
    a = Subspace(2)

    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(Subspace(0, 1), b1)])
    b3 = Subspace([ControlledSubspace(b2, Subspace(0, 2))])
    Permutation(a, b3).verify()

@pytest.mark.xfail
def test_brute_force_double_rotation_right_left():
    a = Subspace(2)

    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(b1, Subspace(0, 1))])
    b3 = Subspace([ControlledSubspace(Subspace(0, 2), b2)])
    Permutation(a, b3).verify()


def test_permute_registers():
    PermuteRegisters(Subspace(1), [0]).verify()
    PermuteRegisters(Subspace(2), [0, 1]).verify()
    PermuteRegisters(Subspace(2), [1, 0]).verify()
    PermuteRegisters(Subspace(3), [0, 1, 2]).verify()
    PermuteRegisters(Subspace(3), [0, 2, 1]).verify()
    PermuteRegisters(Subspace(3), [1, 0, 2]).verify()
    PermuteRegisters(Subspace(3), [1, 2, 0]).verify()
    PermuteRegisters(Subspace(3), [2, 0, 1]).verify()
    PermuteRegisters(Subspace(3), [2, 1, 0]).verify()


def test_find_matching_partitioning():
    assert _find_matching_partitioning(Subspace(0), Subspace(0)) == []
    assert _find_matching_partitioning(Subspace(1), Subspace(1)) == [
        (Subspace(1), Subspace(1))
    ]
    assert _find_matching_partitioning(Subspace(2), Subspace(2)) == [
        (Subspace(1), Subspace(1)),
        (Subspace(1), Subspace(1)),
    ]

    c = ControlledSubspace(Subspace(1), Subspace(1))
    assert _find_matching_partitioning(Subspace([c]), Subspace(2)) == [
        (Subspace(1), Subspace(1)),
        (Subspace(1), Subspace(1)),
    ]
    assert _find_matching_partitioning(Subspace(2), Subspace([c])) == [
        (Subspace(1), Subspace(1)),
        (Subspace(1), Subspace(1)),
    ]
    c = ControlledSubspace(Subspace(1), Subspace(0, 1))
    assert _find_matching_partitioning(Subspace([c]), Subspace([c])) == [
        (Subspace([c]), Subspace([c]))
    ]
    assert _find_matching_partitioning(Subspace([ID, c]), Subspace([c, ID])) == [
        (Subspace([ID, c]), Subspace([c, ID]))
    ]
    assert _find_matching_partitioning(Subspace([ID, c]), Subspace([ID, c])) == [
        (Subspace(1), Subspace(1)),
        (Subspace([c]), Subspace([c])),
    ]
