import pytest

from unitaria.subspace import Subspace, ControlledSubspace, ID, ZeroQubit
from unitaria.nodes.permutation.permutation import _find_matching_partitioning, Permutation, PermuteRegisters
from unitaria.verifier import verify


def test_find_permutation_trivial():
    verify(Permutation(Subspace(registers=0), Subspace(registers=0)))
    verify(Permutation(Subspace(registers=0, zero_qubits=1), Subspace(registers=0, zero_qubits=1)))
    verify(Permutation(Subspace(registers=1), Subspace(registers=1)))
    verify(Permutation(Subspace(registers=1, zero_qubits=1), Subspace(registers=1, zero_qubits=1)))
    verify(Permutation(Subspace(registers=2), Subspace(registers=2)))
    c = ControlledSubspace(Subspace(registers=1), Subspace(registers=0, zero_qubits=1))
    verify(Permutation(Subspace(registers=[c]), Subspace(registers=[c])))

    verify(Permutation(Subspace(registers=0), Subspace(registers=0, zero_qubits=1)))
    verify(Permutation(Subspace(registers=1), Subspace(registers=1, zero_qubits=1)))
    verify(Permutation(Subspace(registers=[c]), Subspace(registers=[c], zero_qubits=1)))


def test_find_permutation_matching_partitioning():
    verify(Permutation(Subspace(registers=[ZeroQubit(), ID]), Subspace(registers=1)))
    verify(Permutation(Subspace(registers=1), Subspace(registers=[ZeroQubit(), ID])))


@pytest.mark.xfail
def test_brute_force_1_simple_rotation():
    a = Subspace(registers=1)
    a1 = Subspace(registers=1, zero_qubits=1)
    b = Subspace(registers=[ControlledSubspace(a, a)])
    c = Subspace(registers=[ControlledSubspace(b, a1)])
    d = Subspace(registers=[ControlledSubspace(a1, b)])
    verify(Permutation(d, c))
    verify(Permutation(c, d))


@pytest.mark.xfail
def test_brute_force_2_simple_rotations():
    a = Subspace(registers=2)

    # Left
    b1 = Subspace(registers=1)
    b2 = Subspace(registers=[ControlledSubspace(b1, Subspace(registers=0, zero_qubits=1))])
    b3 = Subspace(registers=[ControlledSubspace(b2, Subspace(registers=0, zero_qubits=2))])
    verify(Permutation(a, b3))
    verify(Permutation(b3, a))

    # Right
    b1 = Subspace(registers=1)
    b2 = Subspace(registers=[ControlledSubspace(Subspace(registers=0, zero_qubits=1), b1)])
    b3 = Subspace(registers=[ControlledSubspace(Subspace(registers=0, zero_qubits=2), b2)])
    verify(Permutation(a, b3))
    verify(Permutation(b3, a))


@pytest.mark.xfail
def test_brute_force_double_rotation_left_right():
    a = Subspace(registers=2)

    b1 = Subspace(registers=1)
    b2 = Subspace(registers=[ControlledSubspace(Subspace(registers=0, zero_qubits=1), b1)])
    b3 = Subspace(registers=[ControlledSubspace(b2, Subspace(registers=0, zero_qubits=2))])
    verify(Permutation(a, b3))


@pytest.mark.xfail
def test_brute_force_double_rotation_right_left():
    a = Subspace(registers=2)

    b1 = Subspace(registers=1)
    b2 = Subspace(registers=[ControlledSubspace(b1, Subspace(registers=0, zero_qubits=1))])
    b3 = Subspace(registers=[ControlledSubspace(Subspace(registers=0, zero_qubits=2), b2)])
    verify(Permutation(a, b3))


def test_permute_registers():
    verify(PermuteRegisters(Subspace(registers=1), [0]))
    verify(PermuteRegisters(Subspace(registers=2), [0, 1]))
    verify(PermuteRegisters(Subspace(registers=2), [1, 0]))
    verify(PermuteRegisters(Subspace(registers=3), [0, 1, 2]))
    verify(PermuteRegisters(Subspace(registers=3), [0, 2, 1]))
    verify(PermuteRegisters(Subspace(registers=3), [1, 0, 2]))
    verify(PermuteRegisters(Subspace(registers=3), [1, 2, 0]))
    verify(PermuteRegisters(Subspace(registers=3), [2, 0, 1]))
    verify(PermuteRegisters(Subspace(registers=3), [2, 1, 0]))


def test_find_matching_partitioning():
    assert _find_matching_partitioning(Subspace(registers=0), Subspace(registers=0)) == []
    assert _find_matching_partitioning(Subspace(registers=1), Subspace(registers=1)) == [
        (Subspace(registers=1), Subspace(registers=1))
    ]
    assert _find_matching_partitioning(Subspace(registers=2), Subspace(registers=2)) == [
        (Subspace(registers=1), Subspace(registers=1)),
        (Subspace(registers=1), Subspace(registers=1)),
    ]

    c = ControlledSubspace(Subspace(registers=1), Subspace(1))
    assert _find_matching_partitioning(Subspace([c]), Subspace(registers=2)) == [
        (Subspace(registers=1), Subspace(registers=1)),
        (Subspace(registers=1), Subspace(registers=1)),
    ]
    assert _find_matching_partitioning(Subspace(registers=2), Subspace(registers=[c])) == [
        (Subspace(registers=1), Subspace(registers=1)),
        (Subspace(registers=1), Subspace(registers=1)),
    ]
    c = ControlledSubspace(Subspace(registers=1), Subspace(registers=0, zero_qubits=1))
    assert _find_matching_partitioning(Subspace(registers=[c]), Subspace(registers=[c])) == [
        (Subspace(registers=[c]), Subspace(registers=[c]))
    ]
    assert _find_matching_partitioning(Subspace(registers=[ID, c]), Subspace(registers=[c, ID])) == [
        (Subspace(registers=[ID, c]), Subspace(registers=[c, ID]))
    ]
    assert _find_matching_partitioning(Subspace(registers=[ID, c]), Subspace(registers=[ID, c])) == [
        (Subspace(registers=1), Subspace(registers=1)),
        (Subspace(registers=[c]), Subspace(registers=[c])),
    ]
