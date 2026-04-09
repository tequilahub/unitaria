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
    ut.verify(permute(ut.Subspace(), ut.Subspace()))
    ut.verify(permute(ut.Subspace("0"), ut.Subspace("0")))
    ut.verify(permute(ut.Subspace("#"), ut.Subspace("#")))
    ut.verify(permute(ut.Subspace("0#"), ut.Subspace("0#")))
    ut.verify(permute(ut.Subspace("00#"), ut.Subspace("00#")))
    c = ut.Subspace("#") | ut.Subspace("0")
    ut.verify(permute(c, c))

    ut.verify(permute(ut.Subspace(), ut.Subspace("0")))
    ut.verify(permute(ut.Subspace("#"), ut.Subspace("0#")))
    ut.verify(permute(c, ut.Subspace("0") & c))


def test_find_permutation_matching_partitioning():
    ut.verify(permute(ut.Subspace("#0"), ut.Subspace("#")))
    ut.verify(permute(ut.Subspace("#"), ut.Subspace("#0")))


def test_add_zeros_to_control():
    subspace_in = ut.Subspace("00#") | ut.Subspace("000")
    ut.verify(AddZerosToControl(subspace_in, 0))
    ut.verify(AddZerosToControl(subspace_in, 1))
    ut.verify(AddZerosToControl(subspace_in, 2))
    ut.verify(AddZerosToControl.remove_zeros(subspace_in, 1))
    ut.verify(AddZerosToControl.remove_zeros(subspace_in, 2))


def test_subspace_rotation():
    ut.verify(_rotate(ut.Subspace("##"), False))
    ut.verify(_rotate(ut.Subspace("##"), True))
    subspace = ut.Subspace("#") | ut.Subspace("0")
    ut.verify(_rotate(subspace, True))
    subspace = ut.Subspace("0") | ut.Subspace("#")
    ut.verify(_rotate(subspace, False))
    subspace = (ut.Subspace("#0") | (ut.Subspace("0") | ut.Subspace("#"))) & ut.Subspace("#")
    ut.verify(_rotate(subspace, True))
    ut.verify(_rotate(subspace, False))
    subspace = ut.Subspace("0") & (ut.Subspace("#0") | (ut.Subspace("0") | ut.Subspace("#"))) & ut.Subspace("#")
    ut.verify(_rotate(subspace, True))
    ut.verify(_rotate(subspace, False))
    subspace = ut.Subspace("000") & (ut.Subspace("#0") | (ut.Subspace("0") | ut.Subspace("#"))) & ut.Subspace("#")
    ut.verify(_rotate(subspace, True))
    ut.verify(_rotate(subspace, False))


@pytest.mark.parametrize(
    "subspace",
    [
        ut.Subspace("##"),
        ut.Subspace("###"),
        ut.Subspace("0") | ut.Subspace("#"),
        (ut.Subspace("00#") | ut.Subspace("0#0")),
    ],
)
def test_rotate_to(subspace):
    for i in range(1, subspace.dimension):
        ut.verify(_rotate_to(subspace, i))


def test_1_simple_rotation():
    a = ut.Subspace("#")
    a1 = ut.Subspace("0#")
    b = a | a
    c = b | a1
    d = a1 | b
    ut.verify(permute(d, c))
    ut.verify(permute(c, d))


def test_2_simple_rotations():
    a = ut.Subspace("##")

    # Left
    b1 = ut.Subspace("#")
    b2 = b1 | ut.Subspace("0")
    b3 = b2 | ut.Subspace("00")
    ut.verify(permute(a, b3))
    ut.verify(permute(b3, a))

    # Right
    b1 = ut.Subspace("#")
    b2 = ut.Subspace("0") | b1
    b3 = ut.Subspace("00") | b2
    ut.verify(permute(a, b3))
    ut.verify(permute(b3, a))


def test_double_rotation_left_right():
    a = ut.Subspace("##")

    b1 = ut.Subspace("#")
    b2 = ut.Subspace("0") | b1
    b3 = b2 | ut.Subspace("00")
    ut.verify(permute(a, b3))


def test_double_rotation_right_left():
    a = ut.Subspace("##")

    b1 = ut.Subspace("#")
    b2 = b1 | ut.Subspace("0")
    b3 = ut.Subspace("00") | b2
    ut.verify(permute(a, b3))


def test_permute_registers():
    ut.verify(PermuteFactors(ut.Subspace("#"), [0]))
    ut.verify(PermuteFactors(ut.Subspace("##"), [0, 1]))
    ut.verify(PermuteFactors(ut.Subspace("##"), [1, 0]))
    ut.verify(PermuteFactors(ut.Subspace("###"), [0, 1, 2]))
    ut.verify(PermuteFactors(ut.Subspace("###"), [0, 2, 1]))
    ut.verify(PermuteFactors(ut.Subspace("###"), [1, 0, 2]))
    ut.verify(PermuteFactors(ut.Subspace("###"), [1, 2, 0]))
    ut.verify(PermuteFactors(ut.Subspace("###"), [2, 0, 1]))
    ut.verify(PermuteFactors(ut.Subspace("###"), [2, 1, 0]))


def test_find_matching_partitioning():
    assert _find_matching_partitioning(ut.Subspace(), ut.Subspace()) == []
    assert _find_matching_partitioning(ut.Subspace("#"), ut.Subspace("#")) == [(ut.Subspace("#"), ut.Subspace("#"))]
    assert _find_matching_partitioning(ut.Subspace("##"), ut.Subspace("##")) == [
        (ut.Subspace("#"), ut.Subspace("#")),
        (ut.Subspace("#"), ut.Subspace("#")),
    ]

    c = ut.Subspace("#") | ut.Subspace("#")
    assert _find_matching_partitioning(c, ut.Subspace("##")) == [
        (ut.Subspace("#"), ut.Subspace("#")),
        (ut.Subspace("#"), ut.Subspace("#")),
    ]
    assert _find_matching_partitioning(ut.Subspace("##"), c) == [
        (ut.Subspace("#"), ut.Subspace("#")),
        (ut.Subspace("#"), ut.Subspace("#")),
    ]
    c = ut.Subspace("#") | ut.Subspace("0")
    assert _find_matching_partitioning(c, c) == [(c, c)]
    assert _find_matching_partitioning(c & ut.Subspace("#"), ut.Subspace("#") & c) == [
        (c & ut.Subspace("#"), ut.Subspace("#") & c)
    ]
    assert _find_matching_partitioning(c & ut.Subspace("#"), c & ut.Subspace("#")) == [
        (ut.Subspace("#"), ut.Subspace("#")),
        (c, c),
    ]
