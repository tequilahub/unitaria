from bequem.nodes.controlled_ops import BlockDiagonal, Add
from bequem.nodes.identity import Identity
from bequem.nodes.integer_arithmetic import Increment
from bequem.qubit_map import QubitMap


def test_block_diagonal():

    A = Identity(QubitMap(1, 1))
    B = Increment(2)

    D = BlockDiagonal(A, B)
    D.verify()
    D = BlockDiagonal(B, A)
    D.verify()


def test_add():

    A = Identity(QubitMap(1, 0))
    B = Increment(1)

    D = Add(A, B)
    D.verify()
    D = Add(B, A)
    D.verify()
