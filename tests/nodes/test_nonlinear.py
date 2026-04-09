import unitaria as ut
import numpy as np


def test_componentwise_mul():
    ut.verify(ut.ComponentwiseMul(ut.Subspace("#")))
    ut.verify(ut.ComponentwiseMul(ut.Subspace("0#")))
    ut.verify(ut.ComponentwiseMul((ut.Subspace("#") | ut.Subspace("0")) & ut.Subspace("#")))


def test_componentwise_mul_partial():
    ut.verify(ut.ComponentwiseMul(ut.ConstantVector(np.array([2, 1]))))
    ut.verify(
        ut.ComponentwiseMul(ut.ConstantVector(np.array([2, 1])), ut.ConstantVector(np.array([3, 4]))), np.array([6, 4])
    )
