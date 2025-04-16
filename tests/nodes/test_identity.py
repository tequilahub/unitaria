from bequem.nodes.identity import Identity
from bequem.qubit_map import QubitMap


def test_identity():

    qubits = QubitMap(1)
    I = Identity(qubits)

    I.verify()
