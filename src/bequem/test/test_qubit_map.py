import numpy as np
import tequila as tq

from ..qubit_map import QubitMap, Qubit, Controlled, Projection
from ..circuit import Circuit


def test_simplify():
    assert QubitMap([]).simplify() == QubitMap([])
    assert QubitMap([Qubit.ZERO]).simplify() == QubitMap([Qubit.ZERO])
    assert QubitMap([Qubit.ID]).simplify() == QubitMap([Qubit.ID])
    assert QubitMap(
        [Controlled(QubitMap([Qubit.ID]), QubitMap([Qubit.ZERO]))]
    ).simplify() == QubitMap([Controlled(QubitMap([Qubit.ID]), QubitMap([Qubit.ZERO]))])
    circuit = Circuit()
    assert QubitMap([Projection(circuit)]).simplify() == QubitMap([Projection(circuit)])

    assert QubitMap(
        [Controlled(QubitMap([Qubit.ZERO]), QubitMap([Qubit.ZERO]))]
    ).simplify() == QubitMap([Qubit.ZERO, Qubit.ID])
    c = Controlled(QubitMap([Qubit.ZERO]), QubitMap([Qubit.ZERO]))
    assert QubitMap([Controlled(QubitMap([c]), QubitMap([c]))]).simplify() == QubitMap(
        [Qubit.ZERO, Qubit.ID, Qubit.ID]
    )


def test_reduce():
    assert QubitMap([]).reduce() == QubitMap([])
    assert QubitMap([Qubit.ZERO]).reduce() == QubitMap([])
    assert QubitMap([Qubit.ID]).reduce() == QubitMap([Qubit.ID])
    assert QubitMap(
        [Controlled(QubitMap([Qubit.ID]), QubitMap([Qubit.ZERO]))]
    ).reduce() == QubitMap([Controlled(QubitMap([Qubit.ID]), QubitMap([Qubit.ZERO]))])
    circuit = Circuit()
    assert QubitMap([Projection(circuit)]).reduce() == QubitMap([Projection(circuit)])
    assert QubitMap(
        [
            Qubit.ZERO,
            Qubit.ID,
            Qubit.ZERO,
            Qubit.ZERO,
            Projection(circuit),
            Controlled(QubitMap([Qubit.ZERO]), QubitMap([Qubit.ZERO])),
            Qubit.ZERO,
        ]
    ).reduce() == QubitMap(
        [
            Qubit.ID,
            Projection(circuit),
            Controlled(QubitMap([Qubit.ZERO]), QubitMap([Qubit.ZERO])),
        ]
    )


def test_is_all_zeros():
    assert QubitMap([]).is_all_zeros()
    assert QubitMap([Qubit.ZERO]).is_all_zeros()
    assert not QubitMap([Qubit.ID]).is_all_zeros()
    assert not QubitMap(
        [Controlled(QubitMap([Qubit.ID]), QubitMap([Qubit.ZERO]))]
    ).is_all_zeros()
    circuit = Circuit()
    assert not QubitMap([Projection(circuit)]).is_all_zeros()
    assert not QubitMap(
        [
            Qubit.ZERO,
            Qubit.ID,
            Qubit.ZERO,
            Qubit.ZERO,
            Projection(circuit),
            Controlled(QubitMap([Qubit.ZERO]), QubitMap([Qubit.ZERO])),
            Qubit.ZERO,
        ]
    ).is_all_zeros()


def test_basis():
    assert QubitMap([Qubit.ZERO]).test_basis(0)
    assert not QubitMap([Qubit.ZERO]).test_basis(1)

    np.testing.assert_allclose(QubitMap([]).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(QubitMap([Qubit.ZERO]).enumerate_basis(), np.array([0]))
    np.testing.assert_allclose(QubitMap([Qubit.ID]).enumerate_basis(), np.array([0, 1]))
    # TODO
    # np.testing.assert_allclose(QubitMap([Controlled(QubitMap([Qubit.ID]), QubitMap([Qubit.ZERO]))]).enumerate_basis(), np.array([0, 1, 2]))
    # circuit = Circuit()
    # assert not QubitMap([Projection(circuit)]).enumerate_basis()
    np.testing.assert_allclose(
        QubitMap(
            [
                Qubit.ID,
                Qubit.ZERO,
                # Controlled(QubitMap([Qubit.ZERO]), QubitMap([Qubit.ZERO])),
                # Projection(circuit),
            ]
        ).enumerate_basis(),
        np.array([0, 1]),
    )


def test_total_bits():
    assert QubitMap([]).total_bits() == 0
    assert QubitMap([Qubit.ZERO]).total_bits() == 1
    assert QubitMap([Qubit.ID]).total_bits() == 1
    assert (
        QubitMap(
            [Controlled(QubitMap([Qubit.ID]), QubitMap([Qubit.ZERO]))]
        ).total_bits()
        == 2
    )
    circuit = Circuit()
    circuit.tq_circuit += tq.gates.X(target=0)
    circuit.tq_circuit += tq.gates.X(target=1)
    assert QubitMap([Projection(circuit)]).total_bits() == 1
    assert (
        QubitMap(
            [
                Qubit.ZERO,
                Qubit.ID,
                Qubit.ZERO,
                Qubit.ZERO,
                Projection(circuit),
                Controlled(QubitMap([Qubit.ZERO]), QubitMap([Qubit.ZERO])),
                Qubit.ZERO,
            ]
        ).total_bits()
        == 8
    )
