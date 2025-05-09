from bequem.nodes.integer_arithmetic import Increment
from bequem.nodes.identity import Identity
from bequem.subspace import Subspace


def test_add():
    A = Identity(Subspace(1, 1))
    B = Increment(1)

    D = A + B
    D.verify()
    D = B + A
    D.verify()


def test_mul():
    A = Increment(2)

    (A @ A).verify()
