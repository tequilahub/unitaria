import pytest

from unitaria.subspace import Subspace, ControlledSubspace, ID, ZeroQubit
from unitaria.nodes.permutation.permutation import (
    _find_matching_partitioning,
    permute,
    PermuteRegisters,
    _rotate_to,
    _rotate,
    AddZerosToControl,
)
from unitaria.verifier import verify


def test_find_permutation_trivial():
    verify(permute(Subspace(0), Subspace(0)))
    verify(permute(Subspace(0, 1), Subspace(0, 1)))
    verify(permute(Subspace(1), Subspace(1)))
    verify(permute(Subspace(1, 1), Subspace(1, 1)))
    verify(permute(Subspace(1, 2), Subspace(1, 2)))
    c = ControlledSubspace(Subspace(1), Subspace(0, 1))
    verify(permute(Subspace([c]), Subspace([c])))

    verify(permute(Subspace(0), Subspace(0, 1)))
    verify(permute(Subspace(1), Subspace(1, 1)))
    verify(permute(Subspace([c]), Subspace([c], 1)))


def test_find_permutation_matching_partitioning():
    verify(permute(Subspace([ZeroQubit(), ID]), Subspace(1)))
    verify(permute(Subspace(1), Subspace([ZeroQubit(), ID])))


def test_add_zeros_to_control():
    subspace_in = Subspace([ControlledSubspace(Subspace(1, 2), Subspace(0, 3))])
    verify(AddZerosToControl(subspace_in, 0))
    verify(AddZerosToControl(subspace_in, 1))
    verify(AddZerosToControl(subspace_in, 2))
    verify(AddZerosToControl.remove_zeros(subspace_in, 1))
    verify(AddZerosToControl.remove_zeros(subspace_in, 2))


def test_subspace_rotation():
    verify(_rotate(Subspace(2), False))
    verify(_rotate(Subspace(2), True))
    subspace = Subspace([ControlledSubspace(Subspace(1), Subspace(0, 1))])
    verify(_rotate(subspace, True))
    subspace = Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1))])
    verify(_rotate(subspace, False))
    subspace = Subspace(
        [
            ID,
            ControlledSubspace(
                Subspace([ZeroQubit(), ID]),
                Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1))]),
            ),
        ]
    )
    verify(_rotate(subspace, True))
    verify(_rotate(subspace, False))
    subspace = Subspace(
        [
            ID,
            ControlledSubspace(
                Subspace([ZeroQubit(), ID]),
                Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1))]),
            ),
        ],
        1,
    )
    verify(_rotate(subspace, True))
    verify(_rotate(subspace, False))
    subspace = Subspace(
        [
            ID,
            ControlledSubspace(
                Subspace([ZeroQubit(), ID]),
                Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1))]),
            ),
        ],
        3,
    )
    verify(_rotate(subspace, True))
    verify(_rotate(subspace, False))


@pytest.mark.parametrize(
    "subspace",
    [
        Subspace(2),
        Subspace(3),
        Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1))]),
        Subspace(
            [
                ControlledSubspace(
                    Subspace([ID, ZeroQubit(), ZeroQubit()]),
                    Subspace([ZeroQubit(), ID, ZeroQubit()]),
                ),
            ]
        ),
    ],
)
def test_rotate_to(subspace):
    for i in range(1, subspace.dimension):
        verify(_rotate_to(subspace, i))


def test_1_simple_rotation():
    a = Subspace(1)
    a1 = Subspace(1, 1)
    b = Subspace([ControlledSubspace(a, a)])
    c = Subspace([ControlledSubspace(b, a1)])
    d = Subspace([ControlledSubspace(a1, b)])
    verify(permute(d, c))
    verify(permute(c, d))


def test_2_simple_rotations():
    a = Subspace(2)

    # Left
    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(b1, Subspace(0, 1))])
    b3 = Subspace([ControlledSubspace(b2, Subspace(0, 2))])
    verify(permute(a, b3))
    verify(permute(b3, a))

    # Right
    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(Subspace(0, 1), b1)])
    b3 = Subspace([ControlledSubspace(Subspace(0, 2), b2)])
    verify(permute(a, b3))
    verify(permute(b3, a))


def test_double_rotation_left_right():
    a = Subspace(2)

    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(Subspace(0, 1), b1)])
    b3 = Subspace([ControlledSubspace(b2, Subspace(0, 2))])
    verify(permute(a, b3))


def test_double_rotation_right_left():
    a = Subspace(2)

    b1 = Subspace(1)
    b2 = Subspace([ControlledSubspace(b1, Subspace(0, 1))])
    b3 = Subspace([ControlledSubspace(Subspace(0, 2), b2)])
    verify(permute(a, b3))


def test_permute_registers():
    verify(PermuteRegisters(Subspace(1), [0]))
    verify(PermuteRegisters(Subspace(2), [0, 1]))
    verify(PermuteRegisters(Subspace(2), [1, 0]))
    verify(PermuteRegisters(Subspace(3), [0, 1, 2]))
    verify(PermuteRegisters(Subspace(3), [0, 2, 1]))
    verify(PermuteRegisters(Subspace(3), [1, 0, 2]))
    verify(PermuteRegisters(Subspace(3), [1, 2, 0]))
    verify(PermuteRegisters(Subspace(3), [2, 0, 1]))
    verify(PermuteRegisters(Subspace(3), [2, 1, 0]))


def test_find_matching_partitioning():
    assert _find_matching_partitioning(Subspace(0), Subspace(0)) == []
    assert _find_matching_partitioning(Subspace(1), Subspace(1)) == [(Subspace(1), Subspace(1))]
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
    assert _find_matching_partitioning(Subspace([c]), Subspace([c])) == [(Subspace([c]), Subspace([c]))]
    assert _find_matching_partitioning(Subspace([ID, c]), Subspace([c, ID])) == [(Subspace([ID, c]), Subspace([c, ID]))]
    assert _find_matching_partitioning(Subspace([ID, c]), Subspace([ID, c])) == [
        (Subspace(1), Subspace(1)),
        (Subspace([c]), Subspace([c])),
    ]
