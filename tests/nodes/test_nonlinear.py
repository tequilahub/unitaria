from bequem.nodes import ComponentwiseMul
from bequem.qubit_map import QubitMap, Qubit, ID


def test_componentwise_mul():
    ComponentwiseMul(QubitMap(1)).verify()
    ComponentwiseMul(QubitMap(1, 1)).verify()
    ComponentwiseMul(QubitMap([ID, Qubit(QubitMap(1), QubitMap(0, 1))])).verify()
