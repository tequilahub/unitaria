import numpy as np
import tequila as tq

from unitaria.estimator.estimator import Estimator
from unitaria.nodes.node import Node
from unitaria.util import sample_bound

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
    :param count_gates:
        Wether to count the number of gates. May be much slower.
    """

    def __init__(
        self,
        scheme: str = "exact",
        default_precision: float | None = None,
        default_failure_probability: float | None = None,
        seed: np.random.SeedSequence | None = None,
        count_gates: bool = False,
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
        self.gate_count = {}

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

            if self.count_gates:
                circuit = node.circuit()
                # This is actually slightly cheating, since this way the error
                # of the circuit and sampling might add up to be larger than
                # precision, but since we only use it to count the gates, the
                # difference should only be logarithmic.
                compiler = tq.CircuitCompiler.error_correctable_gate_set(normalized_precision)
                compiled = compiler.compile_circuit(circuit._tq_circuit)

                for gate in compiled.gates:
                    name = gate.name.lower()
                    if name == "globalphase":
                        continue
                    if name == "phase":
                        if gate.parameter < 3 * np.pi / 8:
                            name = "t"
                        else:
                            name = "s"
                    if name == "x":
                        if len(gate.control) > 0:
                            name = "cx"
                    self.gate_count[name] = self.gate_count.get(name, 0) + samples

            return np.sqrt(measurement / samples) * normalization
