import unitaria as ut


def test_add():
    A = ut.Identity(ut.Subspace("0#"))
    B = ut.Increment(bits=1)

    D = A + B
    ut.verify(D)
    D = B + A
    ut.verify(D)


def test_mul():
    A = ut.Increment(bits=2)

    ut.verify(A @ A)
