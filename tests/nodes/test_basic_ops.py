import unitaria as ut


def test_tensor():
    A = ut.Increment(bits=1)
    B = ut.Identity(ut.Subspace("#"))

    ut.verify((A & B))
    ut.verify((A & A))
    ut.verify((B & B))
    ut.verify((B & A))
    ut.verify((B & (A & B)))
    ut.verify(((B & A) & B))


def test_scale():
    A = ut.Increment(bits=1)

    ut.verify(1 * A)
    ut.verify((-1) * A)
    ut.verify(ut.Scale(A, 0.5, absolute=True))
    ut.verify(ut.Scale(A, 0.5, remove_efficiency=2.34))
    ut.verify(ut.Scale(A, 0.5, remove_efficiency=2.34, absolute=True))
