from unitaria.nodes.basic.projection import Projection
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_projection():
    verify(Projection(Subspace(bits=1), Subspace(bits=1)))
    verify(Projection(Subspace(bits=2), Subspace(bits=1, zero_qubits=1)))
    verify(Projection(Subspace(bits=1, zero_qubits=1), Subspace(bits=2)))
