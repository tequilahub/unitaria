from bequem.nodes.identity import Identity
from bequem.qubit_map import QubitMap, ID, ZERO


def test_identity():

    qubits = QubitMap([ID, ZERO])
    I = Identity(qubits)

    I.verify()
