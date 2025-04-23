from bequem.nodes.identity import Identity
from bequem.qubit_map import QubitMap


def test_identity():

    I = Identity(QubitMap(1))
    I.verify()
    I = Identity(QubitMap(0))
    I.verify()
