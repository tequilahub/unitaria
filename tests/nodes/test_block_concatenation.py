from unitaria.nodes.basic.block_horizontal import BlockHorizontal
from unitaria.nodes.basic.block_vertical import BlockVertical
from unitaria.nodes.classical.increment import Increment
from unitaria.nodes.basic.identity import Identity
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_block_horizontal():
    A = Identity(subspace=Subspace(bits=1))
    B = Increment(bits=1)
    verify(BlockHorizontal(A, B))
    verify(BlockHorizontal(B, A))


def test_block_vertical():
    A = Identity(subspace=Subspace(bits=1))
    B = Increment(bits=1)
    verify(BlockVertical(A, B))
    verify(BlockVertical(B, A))
