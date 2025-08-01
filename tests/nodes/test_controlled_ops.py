from unitaria.nodes.identity import Identity
from unitaria.nodes.integer_arithmetic.increment import Increment
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_block_diagonal():
    A = Identity(Subspace(2, 1))
    B = Increment(2)

    D = A | B
    verify(D)
    D = B | A
    verify(D)


def test_block_diagonal_with_permutation():
    A = Identity(Subspace(2))
    B = Increment(2)

    D = A | B
    verify(D)
    D = B | A
    verify(D)
