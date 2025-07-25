from bequem.nodes.identity import Identity
from bequem.nodes.integer_arithmetic.increment import Increment
from bequem.subspace.subspace import Subspace
from bequem.verifier import verify
from bequem.nodes.basic_operations.scale import Scale


def test_tensor():
    A = Increment(1)
    B = Identity(Subspace(1, 1))

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
