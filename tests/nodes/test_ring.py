from bequem.nodes.integer_arithmetic import Increment
from bequem.nodes.identity import Identity
from bequem.subspace import Subspace
from bequem.verifier import verify


def test_add():
    A = Identity(Subspace(1, 1))
    B = Increment(1)

    D = A + B
    verify(D)
    D = B + A
    verify(D)


def test_mul():
    A = Increment(2)

    verify(A @ A)
