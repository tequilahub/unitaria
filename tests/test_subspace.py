import numpy as np
import pytest

from unitaria.subspace import Subspace, ControlledSubspace, ID


def test_eq():
    assert Subspace(0) == Subspace(0)
    assert Subspace(1) == Subspace(1)
    assert Subspace(0, 1) == Subspace(0, 1)
    assert Subspace(1, 0) != Subspace(1, 1)
    assert Subspace(1) == Subspace([ID])
    c = ControlledSubspace(Subspace(0), Subspace(0))
    assert Subspace(1) == Subspace([c])
    c = ControlledSubspace(Subspace(1), Subspace(0, 1))
    assert Subspace([c]) == Subspace([c])
    assert Subspace([c]) != Subspace(1)


def test_is_trivial():
    assert Subspace(0).is_trivial()
    assert Subspace(0, 1).is_trivial()
    assert not Subspace(1).is_trivial()
    assert not Subspace([ControlledSubspace(Subspace(1), Subspace(0, 1))]).is_trivial()
    assert not Subspace(
        [
            ID,
            ControlledSubspace(Subspace(0, 1), Subspace(0, 1)),
        ],
        2,
    ).is_trivial()


def test_basis():
    assert Subspace(0, 1).test_basis(0)
    assert not Subspace(0, 1).test_basis(1)

    np.testing.assert_allclose(Subspace(0).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(Subspace(0, 1).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(Subspace(1).enumerate_basis(), np.array([0, 1]))
    np.testing.assert_allclose(
        Subspace([ControlledSubspace(Subspace(1), Subspace(0, 1))]).enumerate_basis(), np.array([0, 1, 2])
    )
    # TODO
    # circuit = Circuit()
    np.testing.assert_allclose(
        Subspace(
            [
                ID,
                # Controlled(QubitMap(0, 1), QubitMap(0, 1)),
            ],
            1,
        ).enumerate_basis(),
        np.array([0, 1]),
    )


def test_total_qubits():
    assert Subspace(0).total_qubits == 0
    assert Subspace(0, 1).total_qubits == 1
    assert Subspace(1).total_qubits == 1
    assert Subspace([ControlledSubspace(Subspace(1), Subspace(0, 1))]).total_qubits == 2
    assert (
        Subspace(
            [
                ID,
                ControlledSubspace(Subspace(0, 1), Subspace(0, 1)),
            ],
            2,
        ).total_qubits
        == 5
    )


@pytest.mark.parametrize(
    "subspace",
    [
        Subspace(0),
        Subspace(1, 1),
        Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))]),
        Subspace(
            [ID, ControlledSubspace(Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1)), ID]), Subspace(1, 2))]
        ),
    ],
)
def test_circuit(subspace: Subspace):
    subspace.verify_circuit()


@pytest.mark.parametrize(
    "subspace",
    [
        Subspace(0),
        Subspace(1, 1),
        Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))]),
        Subspace(
            [ID, ControlledSubspace(Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1)), ID]), Subspace(1, 2))]
        ),
    ],
)
def trailing_zeros(subspace):
    trailing_zeros = subspace.trailing_zeros()
    assert len(subspace.registers) > trailing_zeros
    assert trailing_zeros == len(subspace.registers) or isinstance(
        subspace.registers[-(trailing_zeros + 1)], ControlledSubspace
    )


def test_case_zero():
    assert Subspace(0).case_zero() is None
    assert Subspace(1, 1).case_zero() == Subspace(0)
    assert Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))]).case_zero() == Subspace(2)

    subspace = Subspace(
        [
            ID,
            ControlledSubspace(
                Subspace(0, 2),
                Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1))]),
            ),
        ]
    )
    case_zero = Subspace(
        [
            ID,
        ],
        2,
    )
    assert subspace.case_zero() == case_zero


@pytest.mark.parametrize(
    ("subspace", "expected"),
    [
        (Subspace(0), "<zero qubit subspace>"),
        (Subspace(1, 1),
            "  │\n"
            "1 0\n"
            "  │\n"
            "0 #"
        ),
        (Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))]),
            "  │\n"
            "2 ?─┬─┐\n"
            "    │ │\n"
            "1   # 0\n"
            "  ╔═╩═╛\n"
            "0 #"),
        (
            Subspace(
                [
                    ID,
                    ControlledSubspace(Subspace([ControlledSubspace(Subspace(0, 1), Subspace(1)), ID]), Subspace(1, 2)),
                ]
            ),
            "  │\n"
            "4 ?─┬─────┐\n"
            "    │     │\n"
            "3   #     0\n"
            "    ║     │\n"
            "2   ?─┬─┐ 0\n"
            "      │ │ │\n"
            "1     0 # #\n"
            "  ╔═══╧═╩═╝\n"
            "0 #"
        ),
    ],
)  # fmt: skip
def test_str(subspace: Subspace, expected: str):
    assert str(subspace) == expected
