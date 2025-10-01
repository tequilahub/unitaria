from unitaria.nodes.basic.projection import Projection
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_projection():
    verify(Projection(Subspace(1), Subspace(1)))
    verify(Projection(Subspace(2), Subspace(1, 1)))
    verify(Projection(Subspace(1, 1), Subspace(2)))
