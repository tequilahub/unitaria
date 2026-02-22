from unitaria.nodes.basic.projection import Projection
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_projection():
    verify(Projection(Subspace(registers=1), Subspace(registers=1)))
    verify(Projection(Subspace(registers=2), Subspace(registers=1, zero_qubits=1)))
    verify(Projection(Subspace(registers=1, zero_qubits=1), Subspace(registers=2)))
