import unitaria as ut


def test_block_diagonal():
    A = ut.Identity(ut.Subspace("0##"))
    B = ut.Increment(bits=2)

    D = A | B
    ut.verify(D)
    D = B | A
    ut.verify(D)


def test_block_diagonal_with_permutation():
    A = ut.Identity(ut.Subspace("##"))
    B = ut.Increment(bits=2)

    D = A | B
    ut.verify(D)
    D = B | A
    ut.verify(D)
