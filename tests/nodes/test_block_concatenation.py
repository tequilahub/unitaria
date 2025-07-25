from bequem.nodes.block_concatenation.block_horizontal import BlockHorizontal
from bequem.nodes.block_concatenation.block_vertical import BlockVertical
from bequem.nodes.integer_arithmetic.increment import Increment
from bequem.nodes.identity import Identity
from bequem.subspace import Subspace
from bequem.verifier import verify


def test_block_horizontal():
    A = Identity(Subspace(1))
    B = Increment(1)
    verify(BlockHorizontal(A, B))
    verify(BlockHorizontal(B, A))


def test_block_vertical():
    A = Identity(Subspace(1))
    B = Increment(1)
    verify(BlockVertical(A, B))
    verify(BlockVertical(B, A))
