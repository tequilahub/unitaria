import pytest

from bequem.nodes import Increment, Identity, Add, Scale, Adjoint
from bequem.qubit_map import Subspace


@pytest.mark.parametrize("n", range(1, 5))
def test_laplace(n: int):
    C = Increment(n)
    A = Add(Scale(Identity(Subspace(n)), -2), Scale(Add(C, Adjoint(C)), 1))
    A.verify()
