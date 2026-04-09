import unitaria as ut


def test_projection():
    ut.verify(ut.Projection(ut.Subspace("#"), ut.Subspace("#")))
    ut.verify(ut.Projection(ut.Subspace("##"), ut.Subspace("0#")))
    ut.verify(ut.Projection(ut.Subspace("0#"), ut.Subspace("##")))
