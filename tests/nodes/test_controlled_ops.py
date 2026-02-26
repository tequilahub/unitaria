from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.classical.increment import Increment
from unitaria.subspace import Subspace
from unitaria.verifier import verify


def test_block_diagonal():
    A = Identity(subspace=Subspace(registers=2, zero_qubits=1))
    B = Increment(bits=2)

    D = A | B
    verify(D)
    D = B | A
    verify(D)


def test_block_diagonal_with_permutation():
    A = Identity(subspace=Subspace(registers=2))
    B = Increment(bits=2)

    D = A | B
    verify(D)
    D = B | A
    verify(D)
