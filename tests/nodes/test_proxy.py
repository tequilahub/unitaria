from bequem.nodes.integer_arithmetic import Increment
from bequem.nodes.identity import Identity
from bequem.qubit_map import QubitMap
from bequem.nodes.proxy_node import BlockHorizontal, BlockVertical


def test_add():
    A = Identity(QubitMap(1, 1))
    B = Increment(1)

    D = A + B
    D.verify()
    D = B + A
    D.verify()


def test_mul():
    A = Increment(2)

    (A @ A).verify()


def test_block_horizontal():
    A = Identity(QubitMap(1))
    B = Increment(1)
    BlockHorizontal(A, B).verify()
    BlockHorizontal(B, A).verify()

def test_block_vertical():
    A = Identity(QubitMap(1))
    B = Increment(1)
    BlockVertical(A, B).verify()
    BlockVertical(B, A).verify()
