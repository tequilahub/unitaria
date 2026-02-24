"""
Encodings of matrices, vectors, and operations between them.

The basic type of this library is `Node` representing a node in a computational
graph. Consider, for example, the following matrix product

>>> from unitaria.nodes import Increment, Identity
>>> print(Increment(2) @ Identity(2))
Mul
├── Identity{'subspace': Subspace(2)}
└── Increment{'bits': 2}

Note how the operation is transformed into a tree of three nodes. These nodes
can now be used to compute the result directly

>>> from unitaria.nodes import Increment, Identity
>>> import numpy as np
>>> (Increment(2) @ Identity(2)).toarray().real
array([[0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.]])

Or the circuit, normalization, and subspaces of the block encoding can be obtained
through the corresponding properties of `Node`.

Custom nodes
------------

Most basic operations are implemented already in this library. See `Node` for
implementing custom nodes directly, or `ProxyNode` for implementing custom nodes
by decomposing them into other operations.
"""

from unitaria.nodes.node import Node
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.multilinear_node import MultilinearNode
from unitaria.nodes.abstract_node import AbstractNode

from unitaria.nodes.constants.constant_vector import ConstantVector
from unitaria.nodes.constants.constant_unitary import ConstantUnitary

from unitaria.nodes.basic.identity import Identity
from unitaria.nodes.basic.projection import Projection
from unitaria.nodes.basic.tensor import Tensor
from unitaria.nodes.basic.adjoint import Adjoint
from unitaria.nodes.basic.scale import Scale
from unitaria.nodes.basic.block_diagonal import BlockDiagonal

from unitaria.nodes.basic.mul import Mul
from unitaria.nodes.basic.add import Add
from unitaria.nodes.basic.block_horizontal import BlockHorizontal
from unitaria.nodes.basic.block_vertical import BlockVertical

from unitaria.nodes.nonlinear import ComponentwiseMul

from unitaria.nodes.classical.constant_integer_addition import ConstantIntegerAddition
from unitaria.nodes.classical.constant_integer_multiplication import ConstantIntegerMultiplication
from unitaria.nodes.classical.increment import Increment
from unitaria.nodes.classical.integer_addition import IntegerAddition

from unitaria.nodes.qsvt.qsvt import QSVT

__all__ = [
    "Node",
    "ProxyNode",
    "MultilinearNode",
    "AbstractNode",
    "ConstantVector",
    "ConstantUnitary",
    "Tensor",
    "Adjoint",
    "Scale",
    "BlockDiagonal",
    "Mul",
    "Add",
    "BlockHorizontal",
    "BlockVertical",
    "Identity",
    "Projection",
    "ComponentwiseMul",
    "ConstantIntegerAddition",
    "ConstantIntegerMultiplication",
    "Increment",
    "IntegerAddition",
    "QSVT",
]
