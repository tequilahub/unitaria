from unitaria.nodes.classical.increment import Increment
from unitaria.nodes.basic.identity import Identity
from unitaria.subspace import Subspace
from unitaria.verifier import verify


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
