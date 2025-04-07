from ..qubit_map import QubitMap, Qubit


def test_reduce():
    assert QubitMap([]).reduce() == QubitMap([])
    assert QubitMap([Qubit.ZERO]).reduce() == QubitMap([])
    assert QubitMap([Qubit.ID]).reduce() == QubitMap([Qubit.ID])
    assert QubitMap(
        [Qubit.ZERO, Qubit.ID, Qubit.ZERO, Qubit.ZERO, Qubit.ID, Qubit.ZERO]
    ).reduce() == QubitMap([Qubit.ID, Qubit.ID])
