import numpy as np
import tequila as tq

from bequem.qubit_map import Subspace, ControlledSubspace, ID
from bequem.circuit import Circuit


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
        ], 2
    ).is_trivial()


def test_basis():
    assert Subspace(0, 1).test_basis(0)
    assert not Subspace(0, 1).test_basis(1)

    np.testing.assert_allclose(Subspace(0).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(Subspace(0, 1).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(Subspace(1).enumerate_basis(), np.array([0, 1]))
    np.testing.assert_allclose(Subspace([ControlledSubspace(Subspace(1), Subspace(0, 1))]).enumerate_basis(), np.array([0, 1, 2]))
    # TODO
    # circuit = Circuit()
    np.testing.assert_allclose(
        Subspace(
            [
                ID,
                # Controlled(QubitMap(0, 1), QubitMap(0, 1)),
            ], 1
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
            ], 2
        ).total_qubits
        == 5
    )

