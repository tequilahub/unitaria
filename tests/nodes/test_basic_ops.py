from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.classical.increment import Increment
from unitaria.subspace import Subspace
from unitaria.verifier import verify
from unitaria.nodes.basic.scale import Scale


def test_tensor():
    A = Increment(1)
    B = Identity(Subspace(1))

    verify((A & B))
    verify((A & A))
    verify((B & B))
    verify((B & A))
    verify((B & (A & B)))
    verify(((B & A) & B))


def test_scale():
    A = Increment(1)

    verify(1 * A)
    verify((-1) * A)
    verify(Scale(A, 0.5, absolute=True))
