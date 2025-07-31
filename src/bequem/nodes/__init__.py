"""
Encodings of matrices, vectors, and operations between them.

The basic type of this library is `Node` representing a node in a computational
graph. Consider, for example, the following matrix product

>>> from bequem.nodes import Increment, Identity
>>> print(Increment(2) @ Identity(2))
Mul
├── Identity{'subspace': Subspace(2)}
└── Increment{'bits': 2}

Note how the operation is transformed into a tree of three nodes. These nodes
can now be used to compute the result directly

>>> from bequem.nodes import Increment, Identity
>>> import numpy as np
>>> (Increment(2) @ Identity(2)).toarray().real
array([[0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.]])

Or the circuit, normalization, and subspaces of the block encoding can be obtain
through the corresponding properties of `Node`.

Custom nodes
------------

Most basic operations are implemented already in this library. See `Node` for
implementing custom nodes directly, or `ProxyNode` for implementing custom nodes
by decomposing them into other operations.
"""

from bequem.nodes.node import Node
from bequem.nodes.proxy_node import ProxyNode

from bequem.nodes.constants.constant_vector import ConstantVector
from bequem.nodes.constants.constant_unitary import ConstantUnitary

from bequem.nodes.basic_operations.tensor import Tensor
from bequem.nodes.basic_operations.adjoint import Adjoint
from bequem.nodes.basic_operations.scale import Scale


from bequem.nodes.permutation import Permutation
from bequem.nodes.controlled_operations.block_diagonal import BlockDiagonal


from bequem.nodes.ring_operations.mul import Mul
from bequem.nodes.ring_operations.add import Add


from bequem.nodes.block_concatenation.block_horizontal import BlockHorizontal
from bequem.nodes.block_concatenation.block_vertical import BlockVertical


from bequem.nodes.identity import Identity
from bequem.nodes.nonlinear import ComponentwiseMul


from bequem.nodes.integer_arithmetic.constant_integer_addition import ConstantIntegerAddition
from bequem.nodes.integer_arithmetic.constant_integer_multiplication import ConstantIntegerMultiplication

from bequem.nodes.integer_arithmetic.increment import Increment
from bequem.nodes.integer_arithmetic.integer_addition import IntegerAddition

__all__ = [
    "Node",
    "ProxyNode",
    "ConstantVector",
    "ConstantUnitary",
    "Tensor",
    "Adjoint",
    "Scale",
    "Permutation",
    "BlockDiagonal",
    "Mul",
    "Add",
    "BlockHorizontal",
    "BlockVertical",
    "Identity",
    "ComponentwiseMul",
    "ConstantIntegerAddition",
    "ConstantIntegerMultiplication",
    "Increment",
    "IntegerAddition",
]
