from bequem.nodes.identity import Identity
from bequem.nodes.integer_arithmetic import Increment
from bequem.subspace import Subspace


def test_tensor():
    A = Increment(1)
    B = Identity(Subspace(1, 1))

    (A & B).verify()
    (A & A).verify()
    (B & B).verify()
    (B & A).verify()
    (B & (A & B)).verify()
    ((B & A) & B).verify()
