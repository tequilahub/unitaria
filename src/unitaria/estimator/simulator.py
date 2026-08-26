from collections.abc import Callable

import numpy as np
from scipy.stats import binom

from unitaria.estimator.estimator import Estimator
from unitaria.nodes.node import Node
from unitaria.util import sample_bound
from unitaria.t_count import get_t_count
from unitaria.circuit import Circuit

# TODO: Implement simulation of phase estimation


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
    :param complexity_metric:
        A function which takes a `Circuit` and a precision, and returns a number
        to be used as the complexity for running that circuit. Defaults to an
        estimate of the required T gates. Set to ``None`` to disable complexity
        tracking, which avoids compliation of circuits.
    :param qubits:
        Determines how many ancillae are passed to `Node.circuit`.
        Specifically, the total qubits passed to that function will
        be the maximum of ``qubits`` or the qubits required to
        encode the node.
        If you want to estimate gates for a specific device, set ``qubits``
        to the number of qubits in that device.
    """

    def __init__(
        self,
        scheme: str = "exact",
        default_precision: float | None = None,
        default_failure_probability: float | None = None,
        seed: np.random.SeedSequence | None = None,
        complexity_metric: Callable[[Circuit, float], int] | None = get_t_count,
        qubits: int = 100,
    ):
        if scheme == "exact":
            if default_precision is not None or default_failure_probability is not None:
                raise ValueError('"exact" simulator does not support `precision` or `failure_probability` arguments')
            default_precision = 0
            default_failure_probability = 0
        else:
            if scheme not in ["monte-carlo", "phase-estimation"]:
                raise ValueError('Supported schemes are "exact", "monte-carlo" and "phase-estimation"')
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
        self.complexity_metric = complexity_metric
        self.qubits = qubits
        self.complexity = 0

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

        if self.scheme not in ["monte-carlo", "phase-estimation"]:
            raise ValueError('Supported schemes are "exact", "monte-carlo" and "phase-estimation"')
        if precision <= 0:
            raise ValueError('Simulator with scheme != "exact" requires precision > 0')
        if not 0 < failure_probability < 1:
            raise ValueError('Simulator with scheme != "exact" requires 0 < failure_probability < 1')

        norm = node.compute_norm()
        normalization = node.normalization
        information_efficiency = norm / normalization
        normalized_precision = precision / normalization
        samples = None
        result = None

        if self.scheme == "monte-carlo":
            samples = sample_bound(normalized_precision, failure_probability)
            measurement = self.rng.binomial(samples, information_efficiency**2)
            result = np.sqrt(measurement / samples)
        elif self.scheme == "phase-estimation":
            # Following https://arxiv.org/abs/quant-ph/0005055

            # Find steps such that pi/steps + pi^2/steps^2 <= normalized_precision
            steps = int(
                np.ceil((np.pi + np.sqrt(np.pi**2 + 4 * normalized_precision * np.pi**2)) / (2 * normalized_precision))
            )

            # Compute number of tries using Chernoff bound
            tries = compute_retries(failure_probability)

            theta = np.arcsin(np.sqrt(information_efficiency))

            # Distances
            d = theta / np.pi - np.arange(steps) / steps
            probs = (np.sinc(steps * d) / np.sinc(d)) ** 2
            probs /= probs.sum()

            measured = self.rng.choice(steps, p=probs, size=tries)

            samples = tries * steps
            result = float(np.sin(np.pi * np.median(measured) / steps) ** 2)

        if self.complexity_metric is not None:
            self.count_gates(node, samples=samples)

        return result * normalization

    def count_gates(
        self,
        node: Node,
        precision: float | None = None,
        failure_probability: float | None = None,
        samples: int | None = None,
    ):
        """
        Count the number of gates required to measure the norm of the given block encoding.

        :param node: The node representing the vector of which to compute the norm.
        :param precision:
            The absolute precision, with which the norm should be computed. If
            ``None``, ``self.default_precision`` is used instead.
        :param failure_probability:
            The maximum allowed failure probability, with which the absolute
            error of the estimate may exceed the given precision. If ``None``,
            ``self.default_failure_probability`` is used instead.
        :param samples:
            Number of times that the block encoding is executed. If given,
            overrides the number of samples computed from ``precision`` and
            ``failure_probability``.
        """
        if precision is None:
            precision = self.default_precision
        if failure_probability is None:
            failure_probability = self.default_failure_probability

        if self.scheme == "exact":
            raise ValueError('"exact" simulator does not support gate counting')

        if self.scheme not in ["monte-carlo", "phase-estimation"]:
            raise ValueError('Supported schemes are "exact", "monte-carlo" and "phase-estimation"')
        if precision <= 0:
            raise ValueError('Simulator with scheme != "exact" requires precision > 0')
        if not 0 < failure_probability < 1:
            raise ValueError('Simulator with scheme != "exact" requires 0 < failure_probability < 1')
        normalization = node.normalization
        normalized_precision = precision / normalization

        if samples is None:
            if self.scheme == "monte-carlo":
                samples = sample_bound(normalized_precision, failure_probability)
            elif self.scheme == "phase-estimation":
                # Following https://arxiv.org/abs/quant-ph/0005055
                p0 = 8.0 / np.pi**2

                # Find steps such that pi/steps + pi^2/steps^2 <= normalized_precision
                steps = int(
                    np.ceil(
                        (np.pi + np.sqrt(np.pi**2 + 4 * normalized_precision * np.pi**2)) / (2 * normalized_precision)
                    )
                )

                # Compute number of tries using Chernoff bound
                gap = 2 * p0 - 1
                tries = (
                    1
                    if failure_probability >= 1.0 - p0
                    else max(1, int(np.ceil(2 * np.log(1.0 / failure_probability) / gap**2)))
                )

                samples = tries * steps

        target_qubits = node.subspace_out.total_qubits
        ancilla_count = max(
            self.qubits - target_qubits - node.borrowed_ancilla_count(),
            node.clean_ancilla_count(),
        )
        circuit = node._cached_circuit(ancilla_count, node.borrowed_ancilla_count(), False)

        self.complexity += samples * self.complexity_metric(circuit, normalized_precision)


def compute_retries(target_failure=0.01, p_single_success=8 / (np.pi**2)):
    """
    Computes the minimum odd number of independent runs K such that
    the failure probability of taking the median is <= target_failure.
    """
    K = 1
    while True:
        # Median fails if floor(K/2) + 1 or more runs fail
        max_failures_allowed = K // 2
        p_single_fail = 1.0 - p_single_success

        # Cumulative probability of getting more than (K//2) failures
        p_fail_median = 1.0 - binom.cdf(max_failures_allowed, K, p_single_fail)

        if p_fail_median <= target_failure:
            return K

        K += 2  # Keep K odd to prevent ties in median calculation
