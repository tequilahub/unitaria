from unitaria.nodes.basic.identity import Identity
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_identity():
    Id = Identity(subspace=Subspace(bits=1))
    verify(Id)
    Id = Identity(subspace=Subspace(bits=0))
    verify(Id)
