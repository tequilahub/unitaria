from bequem.nodes.identity import Identity
from bequem.subspace import Subspace
from bequem.verifier import verify


def test_identity():
    Id = Identity(Subspace(1))
    verify(Id)
    Id = Identity(Subspace(0))
    verify(Id)
