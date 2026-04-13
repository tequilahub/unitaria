import unitaria as ut


def test_identity():
    Id = ut.Identity(subspace=ut.Subspace(bits=1))
    ut.verify(Id)
    Id = ut.Identity(subspace=ut.Subspace(bits=0))
    ut.verify(Id)
