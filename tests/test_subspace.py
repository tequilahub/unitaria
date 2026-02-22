import numpy as np
import pytest

from unitaria.subspace import Subspace, ControlledSubspace, ID


def test_eq():
    assert Subspace(registers=0) == Subspace(registers=0)
    assert Subspace(registers=1) == Subspace(registers=1)
    assert Subspace(registers=0, zero_qubits=1) == Subspace(registers=0, zero_qubits=1)
    assert Subspace(registers=1, zero_qubits=0) != Subspace(registers=1, zero_qubits=1)
    assert Subspace(registers=1) == Subspace(registers=[ID])
    c = ControlledSubspace(Subspace(registers=0), Subspace(registers=0))
    assert Subspace(registers=1) == Subspace(registers=[c])
    c = ControlledSubspace(Subspace(registers=1), Subspace(registers=0, zero_qubits=1))
    assert Subspace(registers=[c]) == Subspace(registers=[c])
    assert Subspace(registers=[c]) != Subspace(registers=1)


def test_is_trivial():
    assert Subspace(registers=0).is_trivial()
    assert Subspace(registers=0, zero_qubits=1).is_trivial()
    assert not Subspace(registers=1).is_trivial()
    assert not Subspace(
        registers=[ControlledSubspace(Subspace(registers=1), Subspace(registers=0, zero_qubits=1))]
    ).is_trivial()
    assert not Subspace(
        registers=[
            ID,
            ControlledSubspace(Subspace(registers=0, zero_qubits=1), Subspace(registers=0, zero_qubits=1)),
        ],
        zero_qubits=2,
    ).is_trivial()


def test_basis():
    assert Subspace(registers=0, zero_qubits=1).test_basis(0)
    assert not Subspace(registers=0, zero_qubits=1).test_basis(1)

    np.testing.assert_allclose(Subspace(registers=0).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(Subspace(registers=0, zero_qubits=1).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(Subspace(registers=1).enumerate_basis(), np.array([0, 1]))
    np.testing.assert_allclose(
        Subspace(
            registers=[ControlledSubspace(Subspace(registers=1), Subspace(registers=0, zero_qubits=1))]
        ).enumerate_basis(),
        np.array([0, 1, 2]),
    )
    # TODO
    # circuit = Circuit()
    np.testing.assert_allclose(
        Subspace(
            registers=[
                ID,
                # Controlled(QubitMap(0, 1), QubitMap(0, 1)),
            ],
            zero_qubits=1,
        ).enumerate_basis(),
        np.array([0, 1]),
    )


def test_total_qubits():
    assert Subspace(registers=0).total_qubits == 0
    assert Subspace(registers=0, zero_qubits=1).total_qubits == 1
    assert Subspace(registers=1).total_qubits == 1
    assert (
        Subspace(
            registers=[ControlledSubspace(Subspace(registers=1), Subspace(registers=0, zero_qubits=1))]
        ).total_qubits
        == 2
    )
    assert (
        Subspace(
            registers=[
                ID,
                ControlledSubspace(Subspace(registers=0, zero_qubits=1), Subspace(registers=0, zero_qubits=1)),
            ],
            zero_qubits=2,
        ).total_qubits
        == 5
    )


@pytest.mark.parametrize(
    "subspace",
    [
        Subspace(registers=0),
        Subspace(registers=1, zero_qubits=1),
        Subspace(registers=[ID, ControlledSubspace(Subspace(registers=1), Subspace(registers=0, zero_qubits=1))]),
        Subspace(
            registers=[
                ID,
                ControlledSubspace(
                    Subspace(
                        registers=[ControlledSubspace(Subspace(registers=0, zero_qubits=1), Subspace(registers=1)), ID]
                    ),
                    Subspace(registers=1, zero_qubits=2),
                ),
            ]
        ),
    ],
)
def test_circuit(subspace: Subspace):
    subspace.verify_circuit()
