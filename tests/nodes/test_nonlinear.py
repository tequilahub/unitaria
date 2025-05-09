from bequem.nodes import ComponentwiseMul
from bequem.subspace import Subspace, ControlledSubspace, ID


def test_componentwise_mul():
    ComponentwiseMul(Subspace(1)).verify()
    ComponentwiseMul(Subspace(1, 1)).verify()
    ComponentwiseMul(Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))])).verify()
