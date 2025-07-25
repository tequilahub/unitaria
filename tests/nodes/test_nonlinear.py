from bequem.nodes import ComponentwiseMul
from bequem.subspace import Subspace, ControlledSubspace, ID
from bequem.verifier import verify


def test_componentwise_mul():
    verify(ComponentwiseMul(Subspace(1)))
    verify(ComponentwiseMul(Subspace(1, 1)))
    verify(ComponentwiseMul(Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))])))
