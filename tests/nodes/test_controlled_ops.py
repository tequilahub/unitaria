from bequem.nodes.identity import Identity
from bequem.nodes.integer_arithmetic.increment import Increment
from bequem.subspace import Subspace
from bequem.verifier import verify


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
