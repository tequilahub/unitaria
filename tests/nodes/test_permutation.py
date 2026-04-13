import pytest

import unitaria as ut
from unitaria.nodes.permutation.permutation import (
    _find_matching_partitioning,
    permute,
    PermuteFactors,
    _rotate_to,
    _rotate,
    AddZerosToControl,
)


def test_find_permutation_trivial():
    ut.verify(permute(ut.Subspace(bits=0), ut.Subspace(bits=0)))
    ut.verify(permute(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=0, zero_qubits=1)))
    ut.verify(permute(ut.Subspace(bits=1), ut.Subspace(bits=1)))
    ut.verify(permute(ut.Subspace(bits=1, zero_qubits=1), ut.Subspace(bits=1, zero_qubits=1)))
    ut.verify(permute(ut.Subspace(bits=1, zero_qubits=2), ut.Subspace(bits=1, zero_qubits=2)))
    c = ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))
    ut.verify(permute(ut.Subspace([c]), ut.Subspace([c])))

    ut.verify(permute(ut.Subspace(bits=0), ut.Subspace(bits=0, zero_qubits=1)))
    ut.verify(permute(ut.Subspace(bits=1), ut.Subspace(bits=1, zero_qubits=1)))
    ut.verify(permute(ut.Subspace([c]), ut.Subspace([c], zero_qubits=1)))


def test_find_permutation_matching_partitioning():
    ut.verify(permute(ut.Subspace([ut.ZeroQubit(), ut.ID]), ut.Subspace(bits=1)))
    ut.verify(permute(ut.Subspace(bits=1), ut.Subspace([ut.ZeroQubit(), ut.ID])))


def test_add_zeros_to_control():
    subspace_in = ut.Subspace(
        [ut.ControlledSubspace(ut.Subspace(bits=1, zero_qubits=2), ut.Subspace(bits=0, zero_qubits=3))]
    )
    ut.verify(AddZerosToControl(subspace_in, 0))
    ut.verify(AddZerosToControl(subspace_in, 1))
    ut.verify(AddZerosToControl(subspace_in, 2))
    ut.verify(AddZerosToControl.remove_zeros(subspace_in, 1))
    ut.verify(AddZerosToControl.remove_zeros(subspace_in, 2))


def test_subspace_rotation():
    ut.verify(_rotate(ut.Subspace(bits=2), False))
    ut.verify(_rotate(ut.Subspace(bits=2), True))
    subspace = ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))])
    ut.verify(_rotate(subspace, True))
    subspace = ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1))])
    ut.verify(_rotate(subspace, False))
    subspace = ut.Subspace(
        [
            ut.ID,
            ut.ControlledSubspace(
                ut.Subspace([ut.ZeroQubit(), ut.ID]),
                ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1))]),
            ),
        ]
    )
    ut.verify(_rotate(subspace, True))
    ut.verify(_rotate(subspace, False))
    subspace = ut.Subspace(
        [
            ut.ID,
            ut.ControlledSubspace(
                ut.Subspace([ut.ZeroQubit(), ut.ID]),
                ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1))]),
            ),
        ],
        zero_qubits=1,
    )
    ut.verify(_rotate(subspace, True))
    ut.verify(_rotate(subspace, False))
    subspace = ut.Subspace(
        [
            ut.ID,
            ut.ControlledSubspace(
                ut.Subspace([ut.ZeroQubit(), ut.ID]),
                ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1))]),
            ),
        ],
        zero_qubits=3,
    )
    ut.verify(_rotate(subspace, True))
    ut.verify(_rotate(subspace, False))


@pytest.mark.parametrize(
    "subspace",
    [
        ut.Subspace(bits=2),
        ut.Subspace(bits=3),
        ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1))]),
        ut.Subspace(
            [
                ut.ControlledSubspace(
                    ut.Subspace([ut.ID, ut.ZeroQubit(), ut.ZeroQubit()]),
                    ut.Subspace([ut.ZeroQubit(), ut.ID, ut.ZeroQubit()]),
                ),
            ]
        ),
    ],
)
def test_rotate_to(subspace):
    for i in range(1, subspace.dimension):
        ut.verify(_rotate_to(subspace, i))


def test_1_simple_rotation():
    a = ut.Subspace(bits=1)
    a1 = ut.Subspace(bits=1, zero_qubits=1)
    b = ut.Subspace([ut.ControlledSubspace(a, a)])
    c = ut.Subspace([ut.ControlledSubspace(b, a1)])
    d = ut.Subspace([ut.ControlledSubspace(a1, b)])
    ut.verify(permute(d, c))
    ut.verify(permute(c, d))


def test_2_simple_rotations():
    a = ut.Subspace(bits=2)

    # Left
    b1 = ut.Subspace(bits=1)
    b2 = ut.Subspace([ut.ControlledSubspace(b1, ut.Subspace(bits=0, zero_qubits=1))])
    b3 = ut.Subspace([ut.ControlledSubspace(b2, ut.Subspace(bits=0, zero_qubits=2))])
    ut.verify(permute(a, b3))
    ut.verify(permute(b3, a))

    # Right
    b1 = ut.Subspace(bits=1)
    b2 = ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), b1)])
    b3 = ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=2), b2)])
    ut.verify(permute(a, b3))
    ut.verify(permute(b3, a))


def test_double_rotation_left_right():
    a = ut.Subspace(bits=2)

    b1 = ut.Subspace(bits=1)
    b2 = ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), b1)])
    b3 = ut.Subspace([ut.ControlledSubspace(b2, ut.Subspace(bits=0, zero_qubits=2))])
    ut.verify(permute(a, b3))


def test_double_rotation_right_left():
    a = ut.Subspace(bits=2)

    b1 = ut.Subspace(bits=1)
    b2 = ut.Subspace([ut.ControlledSubspace(b1, ut.Subspace(bits=0, zero_qubits=1))])
    b3 = ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=2), b2)])
    ut.verify(permute(a, b3))


def test_permute_registers():
    ut.verify(PermuteFactors(ut.Subspace(bits=1), [0]))
    ut.verify(PermuteFactors(ut.Subspace(bits=2), [0, 1]))
    ut.verify(PermuteFactors(ut.Subspace(bits=2), [1, 0]))
    ut.verify(PermuteFactors(ut.Subspace(bits=3), [0, 1, 2]))
    ut.verify(PermuteFactors(ut.Subspace(bits=3), [0, 2, 1]))
    ut.verify(PermuteFactors(ut.Subspace(bits=3), [1, 0, 2]))
    ut.verify(PermuteFactors(ut.Subspace(bits=3), [1, 2, 0]))
    ut.verify(PermuteFactors(ut.Subspace(bits=3), [2, 0, 1]))
    ut.verify(PermuteFactors(ut.Subspace(bits=3), [2, 1, 0]))


def test_find_matching_partitioning():
    assert _find_matching_partitioning(ut.Subspace(bits=0), ut.Subspace(bits=0)) == []
    assert _find_matching_partitioning(ut.Subspace(bits=1), ut.Subspace(bits=1)) == [
        (ut.Subspace(bits=1), ut.Subspace(bits=1))
    ]
    assert _find_matching_partitioning(ut.Subspace(bits=2), ut.Subspace(bits=2)) == [
        (ut.Subspace(bits=1), ut.Subspace(bits=1)),
        (ut.Subspace(bits=1), ut.Subspace(bits=1)),
    ]

    c = ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=1))
    assert _find_matching_partitioning(ut.Subspace([c]), ut.Subspace(bits=2)) == [
        (ut.Subspace(bits=1), ut.Subspace(bits=1)),
        (ut.Subspace(bits=1), ut.Subspace(bits=1)),
    ]
    assert _find_matching_partitioning(ut.Subspace(bits=2), ut.Subspace([c])) == [
        (ut.Subspace(bits=1), ut.Subspace(bits=1)),
        (ut.Subspace(bits=1), ut.Subspace(bits=1)),
    ]
    c = ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))
    assert _find_matching_partitioning(ut.Subspace([c]), ut.Subspace([c])) == [(ut.Subspace([c]), ut.Subspace([c]))]
    assert _find_matching_partitioning(ut.Subspace([ut.ID, c]), ut.Subspace([c, ut.ID])) == [
        (ut.Subspace([ut.ID, c]), ut.Subspace([c, ut.ID]))
    ]
    assert _find_matching_partitioning(ut.Subspace([ut.ID, c]), ut.Subspace([ut.ID, c])) == [
        (ut.Subspace(bits=1), ut.Subspace(bits=1)),
        (ut.Subspace([c]), ut.Subspace([c])),
    ]
