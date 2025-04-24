from bequem.nodes.identity import Identity
from bequem.nodes.integer_arithmetic import Increment
from bequem.qubit_map import QubitMap


def test_block_diagonal():
    A = Identity(QubitMap(2, 1))
    B = Increment(2)

    D = A | B
    D.verify()
    D = B | A
    D.verify()


def test_block_diagonal_with_permutation():
    A = Identity(QubitMap(2))
    B = Increment(2)

    D = A | B
    D.verify()
    D = B | A
    D.verify()
