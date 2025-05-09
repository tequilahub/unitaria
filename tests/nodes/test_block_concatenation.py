from bequem.nodes.block_concatenation import BlockHorizontal, BlockVertical
from bequem.nodes.integer_arithmetic import Increment
from bequem.nodes.identity import Identity
from bequem.qubit_map import Subspace


def test_block_horizontal():
    A = Identity(Subspace(1))
    B = Increment(1)
    BlockHorizontal(A, B).verify()
    BlockHorizontal(B, A).verify()

def test_block_vertical():
    A = Identity(Subspace(1))
    B = Increment(1)
    BlockVertical(A, B).verify()
    BlockVertical(B, A).verify()

