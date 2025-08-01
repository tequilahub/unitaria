from unitaria.nodes import ComponentwiseMul, ConstantVector
from unitaria.subspace import Subspace, ControlledSubspace, ID
from unitaria.verifier import verify
import numpy as np


def test_componentwise_mul():
    verify(ComponentwiseMul(Subspace(1)))
    verify(ComponentwiseMul(Subspace(1, 1)))
    verify(ComponentwiseMul(Subspace([ID, ControlledSubspace(Subspace(1), Subspace(0, 1))])))


def test_componentwise_mul_partial():
    verify(ComponentwiseMul(ConstantVector(np.array([2, 1]))))
    verify(ComponentwiseMul(ConstantVector(np.array([2, 1])), ConstantVector(np.array([3, 4]))), np.array([6, 4]))
