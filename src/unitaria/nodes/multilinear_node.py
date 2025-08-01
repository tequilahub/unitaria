from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.node import Node
from unitaria.nodes.basic_operations.tensor import Tensor
from unitaria.nodes.ring_operations.mul import Mul
from unitaria.nodes.identity import Identity
from unitaria.subspace import Subspace


class MultilinearNode(ProxyNode):
    """
    Class for simplified syntax of multilinear node.

    This class helps with implementing convenient syntax for multilinear
    nodes, like `~unitaria.nodes.ComponentwiseMul`. The node itself corresponds
    to the bilinear operator which maps a tensor product of two vectors to
    their componentwise product. This means that for two nodes ``a`` and ``b``
    corresponding to vectors, their componentwise product would have to be
    written as ``ComponentwiseMul(a.subspace_out) @ (a.subspace * b.subspace)``.
    It would be more natural to write ``ComponentwiseMul(a, b)``, or maybe
    ``ComponentwiseMul(a)`` to get a matrix with ``a`` on its diagonal.

    This is what this class enables. See the source for `ComponentwiseMul` for
    how to use this class.
    """

    apply: list[Node | None]
    dimensions_in: list[int]

    def __init__(self, dimensions_in: list[int], dimension_out: int, *apply: Node | None):
        if len(apply) > len(dimensions_in):
            raise ValueError(
                f"{len(apply)} inputs were supplied to {self.__class__.__name__}, but it takes at most {len(dimensions_in)}"
            )
        for i, a in enumerate(apply):
            if a is not None and a.subspace_out.dimension != dimensions_in[i]:
                raise ValueError(
                    f"Dimension of input does not match. Expected {dimensions_in[i]} got {a.subspace_out.dimension}"
                )

        self.dimensions_in = dimensions_in
        self.apply = apply

    def _definition_internal(self) -> Node:
        if self._definition is None:
            apply_product = None
            for i, dim in enumerate(self.dimensions_in):
                if i < len(self.apply) and self.apply[i] is not None:
                    node = self.apply[i]
                else:
                    node = Identity(Subspace.from_dim(dim))
                if apply_product is None:
                    apply_product = node
                else:
                    apply_product = Tensor(apply_product, node)
            definition = self.definition()
            self._definition = Mul(apply_product, definition)
        return self._definition
