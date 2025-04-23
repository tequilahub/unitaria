from bequem.nodes.basic_ops import Tensor
from bequem.nodes.identity import Identity
from bequem.nodes.integer_arithmetic import Increment
from bequem.qubit_map import QubitMap


def test_tensor():
    A = Increment(1)
    B = Identity(QubitMap(1, 1))

    (A & B).verify()
    (A & A).verify()
    (B & B).verify()
    (B & A).verify()
    (B & (A & B)).verify()
    ((B & A) & B).verify()


def test_mul():
    A = Increment(2)

    (A @ A).verify()
