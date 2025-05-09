from bequem.nodes.identity import Identity
from bequem.subspace import Subspace


def test_identity():

    I = Identity(Subspace(1))
    I.verify()
    I = Identity(Subspace(0))
    I.verify()
