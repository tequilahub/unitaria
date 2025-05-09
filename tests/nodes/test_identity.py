from bequem.nodes.identity import Identity
from bequem.qubit_map import Subspace


def test_identity():

    I = Identity(Subspace(1))
    I.verify()
    I = Identity(Subspace(0))
    I.verify()
