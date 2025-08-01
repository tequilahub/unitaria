from unitaria.nodes.block_concatenation.block_horizontal import BlockHorizontal
from unitaria.nodes.block_concatenation.block_vertical import BlockVertical
from unitaria.nodes.integer_arithmetic.increment import Increment
from unitaria.nodes.identity import Identity
from unitaria.subspace import Subspace
from unitaria.verifier import verify


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
