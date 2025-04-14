import numpy as np
import tequila as tq

from bequem.qubit_map import QubitMap, Controlled, Projection, ID, ZERO
from bequem.circuit import Circuit


def test_simplify():
    assert QubitMap([]).simplify() == QubitMap([])
    assert QubitMap([ZERO]).simplify() == QubitMap([ZERO])
    assert QubitMap([ID]).simplify() == QubitMap([ID])
    assert QubitMap(
        [Controlled(QubitMap([ID]), QubitMap([ZERO]))]
    ).simplify() == QubitMap([Controlled(QubitMap([ID]), QubitMap([ZERO]))])
    circuit = Circuit()
    assert QubitMap([Projection(circuit)]).simplify() == QubitMap([Projection(circuit)])

    assert QubitMap(
        [Controlled(QubitMap([ZERO]), QubitMap([ZERO]))]
    ).simplify() == QubitMap([ZERO, ID])
    c = Controlled(QubitMap([ZERO]), QubitMap([ZERO]))
    assert QubitMap([Controlled(QubitMap([c]), QubitMap([c]))]).simplify() == QubitMap(
        [ZERO, ID, ID]
    )


def test_is_all_zeros():
    assert QubitMap([]).is_trivial()
    assert QubitMap([ZERO]).is_trivial()
    assert not QubitMap([ID]).is_trivial()
    assert not QubitMap(
        [Controlled(QubitMap([ID]), QubitMap([ZERO]))]
    ).is_trivial()
    circuit = Circuit()
    assert not QubitMap([Projection(circuit)]).is_trivial()
    assert not QubitMap(
        [
            ZERO,
            ID,
            ZERO,
            ZERO,
            Projection(circuit),
            Controlled(QubitMap([ZERO]), QubitMap([ZERO])),
            ZERO,
        ]
    ).is_trivial()


def test_basis():
    assert QubitMap([ZERO]).test_basis(0)
    assert not QubitMap([ZERO]).test_basis(1)

    np.testing.assert_allclose(QubitMap([]).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(QubitMap([ZERO]).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(QubitMap([ID]).enumerate_basis(), np.array([0, 1]))
    # TODO
    # np.testing.assert_allclose(QubitMap([Controlled(QubitMap([ID]), QubitMap([ZERO]))]).enumerate_basis(), np.array([0, 1, 2]))
    # circuit = Circuit()
    # assert not QubitMap([Projection(circuit)]).enumerate_basis()
    np.testing.assert_allclose(
        QubitMap(
            [
                ID,
                ZERO,
                # Controlled(QubitMap([ZERO]), QubitMap([ZERO])),
                # Projection(circuit),
            ]
        ).enumerate_basis(),
        np.array([0, 1]),
    )


def test_total_qubits():
    assert QubitMap([]).total_qubits == 0
    assert QubitMap([ZERO]).total_qubits == 1
    assert QubitMap([ID]).total_qubits == 1
    assert (
        QubitMap(
            [Controlled(QubitMap([ID]), QubitMap([ZERO]))]
        ).total_qubits
        == 2
    )
    circuit = Circuit()
    circuit.tq_circuit += tq.gates.X(target=0)
    circuit.tq_circuit += tq.gates.X(target=1)
    assert QubitMap([Projection(circuit)]).total_qubits == 1
    assert (
        QubitMap(
            [
                ZERO,
                ID,
                ZERO,
                ZERO,
                Projection(circuit),
                Controlled(QubitMap([ZERO]), QubitMap([ZERO])),
                ZERO,
            ]
        ).total_qubits
        == 8
    )
