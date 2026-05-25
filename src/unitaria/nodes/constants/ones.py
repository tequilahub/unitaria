import numpy as np
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.node import Node
from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.basic.block_vertical import BlockVertical
from unitaria.util import logreduce


class Ones(ProxyNode):
    """
    Node representing an all one vector

    :param dim: The dimension of the vector
    """

    def __init__(self, dim: int):
        super().__init__(1, dim)
        if dim <= 0:
            raise ValueError("Dimension must be positive")
        self.dim = dim

    def children(self):
        return []

    def parameters(self):
        return {"dim": self.dim}

    def definition(self) -> Node:
        dim = self.dim

        result = []

        ones_2 = ConstantVector(np.array([1, 1]))

        while dim > 0:
            bits = int(np.floor(np.log2(dim)))
            dim -= 2**bits
            if bits == 0:
                one_bits = ConstantVector(np.array([1]))
            else:
                one_bits = logreduce(lambda x, y: x & y, [ones_2] * bits)

            result.append(one_bits)

        return logreduce(BlockVertical, result)

    def compute(self, input: np.ndarray) -> np.ndarray:
        if input.ndim == 1:
            return np.ones(self.dim) * input[0]
        else:
            return (np.ones((self.dim, 1)) @ input.T).T

    def compute_adjoint(self, input: np.ndarray) -> np.ndarray:
        return (np.ones((1, self.dim)) @ input.T).T
