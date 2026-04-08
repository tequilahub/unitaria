import unitaria as ut


def test_block_horizontal():
    A = ut.Identity(subspace=ut.Subspace(bits=1))
    B = ut.Increment(bits=1)
    ut.verify(ut.BlockHorizontal(A, B))
    ut.verify(ut.BlockHorizontal(B, A))


def test_block_vertical():
    A = ut.Identity(subspace=ut.Subspace(bits=1))
    B = ut.Increment(bits=1)
    ut.verify(ut.BlockVertical(A, B))
    ut.verify(ut.BlockVertical(B, A))
