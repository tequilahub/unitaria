import numpy as np
import tequila as tq

from tequila import BitNumbering

from unitaria.estimator.estimator import Estimator
from unitaria.nodes.node import Node
from unitaria.util import sample_bound


class BackendEstimator(Estimator):
    """
    Estimator based on a simulated or hardware backend.


    :param default_precision:
        Default for the ``precision`` parameter in
        `~BackendEstiator.estimate_norm`.
    :param default_failure_probability:
        Default for the ``failure_probability`` parameter in
        `~BackendEstimator.estimate_norm`.
    :param qubits:
        Determines how many ancillae are passed to `Node.circuit`. Specifically,
        the total qubits passed to that function will be the maximum of
        ``qubits`` or the qubits required to encode the node. If you want to
        run circuits for a specific device, set ``qubits`` to the number of
        qubits in that device.
    :param backend_kwargs:
        Parameters of the backend.
        See :external:py:func:`~tequila.simulators.simulator_api.simulate`.
    """

    def __init__(
        self, default_precision: float, default_failure_probability: float = 0.01, qubits: int = 0, **backend_kwargs
    ):
        super().__init__(default_precision, default_failure_probability)
        self.qubits = qubits
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

        target_qubits = node.subspace_out.total_qubits
        ancilla_count = max(
            self.qubits - target_qubits - node.borrowed_ancilla_count(),
            node.clean_ancilla_count(),
        )
        circuit = node._cached_circuit(ancilla_count, node.borrowed_ancilla_count(), False)
        print(circuit)

        # tq.simulate may also run circuits on a hardware backend
        result = tq.simulate(circuit._tq_circuit, samples=samples, **self.backend_kwargs)
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
