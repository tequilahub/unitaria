from bequem.nodes.block_concatenation import BlockHorizontal, BlockVertical
from bequem.nodes.integer_arithmetic import Increment
from bequem.nodes.identity import Identity
from bequem.qubit_map import QubitMap


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

