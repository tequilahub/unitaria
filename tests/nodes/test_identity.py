from unitaria.nodes.basic.identity import Identity
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_identity():
    Id = Identity(Subspace(1))
    verify(Id)
    Id = Identity(Subspace(0))
    verify(Id)
