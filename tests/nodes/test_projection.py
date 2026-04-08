import unitaria as ut


def test_projection():
    ut.verify(ut.Projection(ut.Subspace(bits=1), ut.Subspace(bits=1)))
    ut.verify(ut.Projection(ut.Subspace(bits=2), ut.Subspace(bits=1, zero_qubits=1)))
    ut.verify(ut.Projection(ut.Subspace(bits=1, zero_qubits=1), ut.Subspace(bits=2)))
