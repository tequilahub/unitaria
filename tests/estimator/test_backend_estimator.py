import unitaria as ut
import numpy as np


def test_qulacs():
    rng = np.random.default_rng(0)
    for precision in [0.1, 0.01]:
        estimator = ut.BackendEstimator(precision, backend="qulacs")
        for i in range(1, 3):
            n = rng.integers(1, 2**i)
            v = rng.standard_normal(2 * n)
            node = ut.ConstantVector(v)
            assert np.abs(estimator.estimate_norm(node[:n]) - np.linalg.norm(v[:n])) < precision
