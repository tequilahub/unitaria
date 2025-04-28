import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Qubit, Projection, ID
from bequem.circuit import Circuit


def test_eq():
    assert QubitMap(0) == QubitMap(0)
    assert QubitMap(1) == QubitMap(1)
    assert QubitMap(0, 1) == QubitMap(0, 1)
    assert QubitMap(1, 0) != QubitMap(1, 1)
    assert QubitMap(1) == QubitMap([ID])
    c = Qubit(QubitMap(0), QubitMap(0))
    assert QubitMap(1) == QubitMap([c])
    c = Qubit(QubitMap(1), QubitMap(0, 1))
    assert QubitMap([c]) == QubitMap([c])
    assert QubitMap([c]) != QubitMap(1)


def test_is_trivial():
    assert QubitMap(0).is_trivial()
    assert QubitMap(0, 1).is_trivial()
    assert not QubitMap(1).is_trivial()
    assert not QubitMap([Qubit(QubitMap(1), QubitMap(0, 1))]).is_trivial()
    circuit = Circuit()
    circuit.tq_circuit.n_qubits = 2
    assert not QubitMap([Projection(circuit, 2)]).is_trivial()
    assert not QubitMap(
        [
            ID,
            Projection(circuit, 2),
            Qubit(QubitMap(0, 1), QubitMap(0, 1)),
        ], 2
    ).is_trivial()


def test_basis():
    assert QubitMap(0, 1).test_basis(0)
    assert not QubitMap(0, 1).test_basis(1)

    np.testing.assert_allclose(QubitMap(0).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(QubitMap(0, 1).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(QubitMap(1).enumerate_basis(), np.array([0, 1]))
    np.testing.assert_allclose(QubitMap([Qubit(QubitMap(1), QubitMap(0, 1))]).enumerate_basis(), np.array([0, 1, 2]))
    # TODO
    # circuit = Circuit()
    # assert not QubitMap([Projection(circuit)]).enumerate_basis()
    np.testing.assert_allclose(
        QubitMap(
            [
                ID,
                # Controlled(QubitMap(0, 1), QubitMap(0, 1)),
                # Projection(circuit),
            ], 1
        ).enumerate_basis(),
        np.array([0, 1]),
    )


def test_total_qubits():
    assert QubitMap(0).total_qubits == 0
    assert QubitMap(0, 1).total_qubits == 1
    assert QubitMap(1).total_qubits == 1
    assert QubitMap([Qubit(QubitMap(1), QubitMap(0, 1))]).total_qubits == 2
    circuit = Circuit()
    circuit.tq_circuit += tq.gates.X(target=1)
    circuit.tq_circuit.n_qubits = 2
    assert QubitMap([Projection(circuit, 0)]).total_qubits == 1
    assert (
        QubitMap(
            [
                ID,
                Projection(circuit, 0),
                Qubit(QubitMap(0, 1), QubitMap(0, 1)),
            ], 2
        ).total_qubits
        == 6
    )

