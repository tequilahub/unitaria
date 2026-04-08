import numpy as np
import pytest
import unitaria as ut


def test_eq():
    assert ut.Subspace(bits=0) == ut.Subspace(bits=0)
    assert ut.Subspace(bits=1) == ut.Subspace(bits=1)
    assert ut.Subspace(bits=0, zero_qubits=1) == ut.Subspace(bits=0, zero_qubits=1)
    assert ut.Subspace(bits=1, zero_qubits=0) != ut.Subspace(bits=1, zero_qubits=1)
    assert ut.Subspace(bits=1) == ut.Subspace([ut.ID])
    c = ut.ControlledSubspace(ut.Subspace(bits=0), ut.Subspace(bits=0))
    assert ut.Subspace(bits=1) == ut.Subspace([c])
    c = ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))
    assert ut.Subspace([c]) == ut.Subspace([c])
    assert ut.Subspace([c]) != ut.Subspace(bits=1)


def test_is_trivial():
    assert ut.Subspace(bits=0).is_trivial()
    assert ut.Subspace(bits=0, zero_qubits=1).is_trivial()
    assert not ut.Subspace(bits=1).is_trivial()
    assert not ut.Subspace(
        [ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))]
    ).is_trivial()
    assert not ut.Subspace(
        [
            ut.ID,
            ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=0, zero_qubits=1)),
        ],
        zero_qubits=2,
    ).is_trivial()


def test_basis():
    assert ut.Subspace(bits=0, zero_qubits=1).test_basis(0)
    assert not ut.Subspace(bits=0, zero_qubits=1).test_basis(1)

    np.testing.assert_allclose(ut.Subspace(bits=0).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(ut.Subspace(bits=0, zero_qubits=1).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(ut.Subspace(bits=1).enumerate_basis(), np.array([0, 1]))
    np.testing.assert_allclose(
        ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))]).enumerate_basis(),
        np.array([0, 1, 2]),
    )
    # TODO
    # circuit = Circuit()
    np.testing.assert_allclose(
        ut.Subspace(
            [
                ut.ID,
                # Controlled(QubitMap(0, 1), QubitMap(0, 1)),
            ],
            zero_qubits=1,
        ).enumerate_basis(),
        np.array([0, 1]),
    )


def test_total_qubits():
    assert ut.Subspace(bits=0).total_qubits == 0
    assert ut.Subspace(bits=0, zero_qubits=1).total_qubits == 1
    assert ut.Subspace(bits=1).total_qubits == 1
    assert (
        ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))]).total_qubits == 2
    )
    assert (
        ut.Subspace(
            [
                ut.ID,
                ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=0, zero_qubits=1)),
            ],
            zero_qubits=2,
        ).total_qubits
        == 5
    )


@pytest.mark.parametrize(
    "subspace",
    [
        ut.Subspace(bits=0),
        ut.Subspace(bits=1, zero_qubits=1),
        ut.Subspace([ut.ID, ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))]),
        ut.Subspace(
            [
                ut.ID,
                ut.ControlledSubspace(
                    ut.Subspace(
                        [
                            ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1)),
                            ut.ID,
                        ]
                    ),
                    ut.Subspace(bits=1, zero_qubits=2),
                ),
            ]
        ),
    ],
)
def test_circuit(subspace: ut.Subspace):
    subspace.verify_circuit()


@pytest.mark.parametrize(
    "subspace",
    [
        ut.Subspace(bits=0),
        ut.Subspace(bits=1, zero_qubits=1),
        ut.Subspace([ut.ID, ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))]),
        ut.Subspace(
            [
                ut.ID,
                ut.ControlledSubspace(
                    ut.Subspace(
                        [ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1)), ut.ID]
                    ),
                    ut.Subspace(bits=1, zero_qubits=2),
                ),
            ]
        ),
    ],
)
def test_trailing_zeros(subspace):
    trailing_zeros = subspace.trailing_zeros()
    assert len(subspace.tensor_factors) >= trailing_zeros
    assert trailing_zeros == len(subspace.tensor_factors) or isinstance(
        subspace.tensor_factors[-(trailing_zeros + 1)], ut.ControlledSubspace
    )


def test_case_zero():
    assert ut.Subspace(bits=0).case_zero() is None
    assert ut.Subspace(bits=1, zero_qubits=1).case_zero() == ut.Subspace(bits=0)
    assert ut.Subspace(
        [ut.ID, ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))]
    ).case_zero() == ut.Subspace(bits=2)

    subspace = ut.Subspace(
        [
            ut.ID,
            ut.ControlledSubspace(
                ut.Subspace(bits=0, zero_qubits=2),
                ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1))]),
            ),
        ]
    )
    case_zero = ut.Subspace(
        bits=1,
        zero_qubits=2,
    )
    assert subspace.case_zero() == case_zero


@pytest.mark.parametrize(
    ("subspace", "expected"),
    [
        (ut.Subspace(bits=0), "<zero qubit subspace>"),
        (ut.Subspace(bits=1, zero_qubits=1),
            "  │\n"
            "1 0\n"
            "  │\n"
            "0 #"
        ),
        (ut.Subspace([ut.ID, ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))]),
            "  │\n"
            "2 ?─┬─┐\n"
            "    │ │\n"
            "1   # 0\n"
            "  ╔═╩═╛\n"
            "0 #"),
        (
            ut.Subspace(
                [
                    ut.ID,
                    ut.ControlledSubspace(ut.Subspace([ut.ControlledSubspace(ut.Subspace(bits=0, zero_qubits=1), ut.Subspace(bits=1)), ut.ID]), ut.Subspace(bits=1, zero_qubits=2)),
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
def test_str(subspace: ut.Subspace, expected: str):
    assert str(subspace) == expected
