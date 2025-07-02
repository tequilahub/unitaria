from bequem.nodes.identity import Identity
from bequem.subspace import Subspace
from bequem.verifier import verify


def test_identity():

    I = Identity(Subspace(1))
    verify(I)
    I = Identity(Subspace(0))
    verify(I)
