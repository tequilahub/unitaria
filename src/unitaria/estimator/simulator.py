from typing import Callable

import numpy as np

from unitaria.estimator.estimator import Estimator
from unitaria.nodes.node import Node
from unitaria.circuit import Circuit
from unitaria.util import sample_bound

# TODO: Implement simulation of phase estimation


def default_count_gates(old_gate_count: int, circuit: Circuit, factor: int) -> int:
    if old_gate_count is None:
        old_gate_count = 0
    return old_gate_count + len(circuit._tq_circuit.gates) * factor


class Simulator(Estimator):
    """
    Estimator based on matrix-arithmetic simulation

    :param scheme:
        The measurement scheme, which is simulated. Should be one of ``"exact"`` or ``"monte-carlo"``.
    :param default_precision:
        Default for the ``precision`` parameter in `~Simulator.estimate_norm`.
    :param default_failure_probability:
        Default for the ``failure_probability`` parameter in
        `~Simulator.estimate_norm`.
    :param count_gates:
        Set to ``False`` to disable gate counting.
        Alternatively, a function may be supplied, which obtains (in this order) the current "gate count" (starting with ``None``), a circuit, and an integer indicating the number of times the circuit is run and should compute the new gate count. The default implementation simply counts the number of gates in the circuit.
    """

    def __init__(
        self,
        scheme: str = "exact",
        default_precision: float | None = None,
        default_failure_probability: float | None = None,
        seed: np.random.SeedSequence | None = None,
        count_gates: Callable | None = default_count_gates,
    ):
        if scheme == "exact":
            if default_precision is not None or default_failure_probability is not None:
                raise ValueError('"exact" simulator does not support `precision` or `failure_probability` arguments')
            default_precision = 0
            default_failure_probability = 0
        else:
            if scheme not in ["monte-carlo"]:
                raise ValueError('Supported schemes are "exact" and "monte-carlo"')
            if default_precision is None:
                raise ValueError('Simulator with scheme != "exact" requires `default_precision` argument')
            if default_failure_probability is None:
                default_failure_probability = 0.01
        super().__init__(default_precision, default_failure_probability)
        self.scheme = scheme
        if seed is None:
            seed = np.random.SeedSequence()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.count_gates = count_gates
        self.gate_count = None

    def estimate_norm(
        self, node: Node, precision: float | None = None, failure_probability: float | None = None
    ) -> float:
        if not node.is_vector():
            raise ValueError("Can only estimate the norm of vectors")

        if precision is None:
            precision = self.default_precision
        if failure_probability is None:
            failure_probability = self.default_failure_probability

        if self.scheme == "exact":
            if precision != 0 or failure_probability != 0:
                raise ValueError('"exact" simulator does not support `precision` or `failure_probability` arguments')
            return node.compute_norm()

        if self.scheme not in ["monte-carlo"]:
            raise ValueError('Supported schemes are "exact" and "monte-carlo"')
        if precision <= 0:
            raise ValueError('Simulator with scheme != "exact" requires precision > 0')
        if not 0 < failure_probability < 1:
            raise ValueError('Simulator with scheme != "exact" requires 0 < failure_probability < 1')

        norm = node.compute_norm()
        normalization = node.normalization
        information_efficiency = norm / normalization
        normalized_precision = precision / normalization

        if self.scheme == "monte-carlo":
            samples = sample_bound(normalized_precision, failure_probability)
            measurement = self.rng.binomial(samples, information_efficiency**2)

            if self.count_gates is not None:
                self.gate_count = self.count_gates(self.gate_count, node.circuit(), samples)

            return np.sqrt(measurement / samples) * normalization
