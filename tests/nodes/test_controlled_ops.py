import unitaria as ut


def test_block_diagonal():
    A = ut.Identity(subspace=ut.Subspace(bits=2, zero_qubits=1))
    B = ut.Increment(bits=2)

    D = A | B
    ut.verify(D)
    D = B | A
    ut.verify(D)


def test_block_diagonal_with_permutation():
    A = ut.Identity(subspace=ut.Subspace(bits=2))
    B = ut.Increment(bits=2)

    D = A | B
    ut.verify(D)
    D = B | A
    ut.verify(D)
