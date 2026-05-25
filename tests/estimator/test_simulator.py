import unitaria as ut
import numpy as np


def test_exact():
    rng = np.random.default_rng(0)
    simulator = ut.Simulator()
    for i in range(1, 3):
        n = rng.integers(1, 2**i)
        v = rng.standard_normal(2 * n)
        node = ut.ConstantVector(v)
        assert np.isclose(simulator.estimate_norm(node[:n]), np.linalg.norm(v[:n]))


def test_monte_carlo():
    rng = np.random.default_rng(0)
    for precision in [0.1, 0.01]:
        simulator = ut.Simulator("monte-carlo", precision)
        old_gate_count = 0
        for i in range(1, 3):
            n = rng.integers(1, 2**i)
            v = rng.standard_normal(2 * n)
            node = ut.ConstantVector(v)
            assert np.abs(simulator.estimate_norm(node[:n]) - np.linalg.norm(v[:n])) < precision
            new_gate_count = simulator.gate_count
            assert (new_gate_count - old_gate_count) > 2 * n * (node.normalization / precision) ** 2
            old_gate_count = new_gate_count
