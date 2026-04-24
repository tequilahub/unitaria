import pytest

import unitaria as ut


@pytest.mark.parametrize("n", range(1, 5))
def test_laplace(n: int):
    C = ut.Increment(bits=n)
    A = ut.Add(
        ut.Scale(ut.Identity(ut.Subspace("#" * n)), -2),
        ut.Scale(ut.Add(C, ut.Adjoint(C)), 1),
    )
    ut.verify(A)
