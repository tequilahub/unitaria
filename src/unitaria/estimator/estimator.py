from abc import ABC, abstractmethod

from unitaria.nodes.node import Node


class Estimator(ABC):
    """
    Interface representing a way to estimate the norm of a block encoding.

    :param default_precision:
        Default for the ``precision`` parameter in `~Estimator.estimate_norm`.
    :param default_failure_probability:
        Default for the ``failure_probability`` parameter in
        `~Estimator.estimate_norm`.
    """

    def __init__(self, default_precision: float, default_failure_probability: float = 0.01):
        self.default_precision = default_precision
        self.default_failure_probability = default_failure_probability

    @abstractmethod
    def estimate_norm(node: Node, precision: float | None = None, failure_probability: float | None = None) -> float:
        """
        Estimate the norm of the given block encoding.

        The block encoding must represent a vector. The returned value is
        guaranteed to lie in the range ``[0, node.normalization]``.

        The implementors of this method are free to interpret the ``precision``argument loosely, and ignore the ``failure_probability`` argument.

        :param node: The node representing the vector of which to compute the norm.
        :param precision:
            The absolute precision, with which the norm should be computed. If
            ``None``, ``self.default_precision`` is used instead.
        :param failure_probability:
            The maximum allowed failure probability, with which the absolute
            error of the estimate may exceed the given precision. If ``None``,
            ``self.default_failure_probability`` is used instead.
        """
        raise NotImplementedError
