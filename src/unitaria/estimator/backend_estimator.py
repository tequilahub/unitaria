import numpy as np
import tequila as tq

from tequila import BitNumbering

from unitaria.estimator.estimator import Estimator
from unitaria.nodes.node import Node
from unitaria.util import sample_bound


class BackendEstimator(Estimator):
    """
    Estimator based on a simulated or hardware backend.

    See :external:py:func:`~tequila.simulators.simulator_api.simulate`.
    """

    def __init__(self, default_precision: float, default_failure_probability: float = 0.01, **backend_kwargs):
        super().__init__(default_precision, default_failure_probability)
        self.backend_kwargs = backend_kwargs

    def estimate_norm(
        self, node: Node, precision: float | None = None, failure_probability: float | None = None
    ) -> float:
        if not node.is_vector():
            raise ValueError("Can only estimate the norm of vectors")
        if precision is None:
            precision = self.default_precision
        if failure_probability is None:
            failure_probability = self.default_failure_probability

        normalization = node.normalization
        normalized_precision = precision / normalization

        samples = sample_bound(normalized_precision, failure_probability)

        circuit = node.circuit()

        result = tq.simulate(circuit._padded(), samples=samples, **self.backend_kwargs)
        measurement = 0
        successful_samples = 0
        max_result = 2**node.subspace_out.total_qubits - 1
        for k, v in result.items():
            k_int = k.to_integer(BitNumbering.LSB)
            if k_int > max_result:
                continue
            successful_samples += v
            if node.subspace_out.test_basis(k_int):
                measurement += v

        return np.sqrt(measurement / successful_samples) * normalization
