import unitaria as ut
import numpy as np


def test_componentwise_mul():
    ut.verify(ut.ComponentwiseMul(ut.Subspace(bits=1)))
    ut.verify(ut.ComponentwiseMul(ut.Subspace(bits=1, zero_qubits=1)))
    ut.verify(
        ut.ComponentwiseMul(
            ut.Subspace([ut.ID, ut.ControlledSubspace(ut.Subspace(bits=1), ut.Subspace(bits=0, zero_qubits=1))])
        )
    )


def test_componentwise_mul_partial():
    ut.verify(ut.ComponentwiseMul(ut.ConstantVector(np.array([2, 1]))))
    ut.verify(
        ut.ComponentwiseMul(ut.ConstantVector(np.array([2, 1])), ut.ConstantVector(np.array([3, 4]))), np.array([6, 4])
    )
