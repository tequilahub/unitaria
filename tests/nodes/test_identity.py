import unitaria as ut


def test_identity():
    Id = ut.Identity(ut.Subspace("#"))
    ut.verify(Id)
    Id = ut.Identity(ut.Subspace())
    ut.verify(Id)
