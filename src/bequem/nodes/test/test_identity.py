from ..identity import Identity
from bequem.qubit_map import QubitMap, Qubit


def test_identity():

    qubits = QubitMap([Qubit.ID, Qubit.ZERO])
    I = Identity(qubits)

    I.verify()
