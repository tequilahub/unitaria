API Reference
=============

Bequem is a library for working with so called "block encodings" of matrices
and vectors. These are format for performing linear algebra calculations on
quantum computers; see also this :doc:`introduction <block_encodings>` and the
:doc:`references`, which contains a list of paper references.

The basic object of this library is a `~bequem.nodes.Node`, which may refer to a
matrix or a vector.

>>> from bequem.nodes import ConstantVector, Identity
>>> Identity(1)
Identity('subspace': Subspace(1))
>>> import numpy as np
>>> ConstantVector(np.array([1, 2]))
ConstantVector('vec': array([1, 2]))

As you can see, nodes are not executed immediately. After all the operations
should be executed on a quantum computer. Instead, you can combine nodes to form
more complex expressions, stored as a computational graph.

>>> from bequem.nodes import ConstantVector, Identity
>>> import numpy as np
>>> print(Identity(1) @ ConstantVector(np.array([1, 2])))
Mul
├── ConstantVector{'vec': array([1, 2])}
└── Identity{'subspace': Subspace(1)}

For a complete list of all available nodes, see `~bequem.nodes`.

There is now two ways in which you can obtain the encoded vector of the above
example. Each node contains a classical implementation of the operation it
performs, available through the method `~bequem.nodes.Node.compute`.

>>> from bequem.nodes import ConstantVector, Identity
>>> import numpy as np
>>> (Identity(1) @ ConstantVector(np.array([1, 2])))).compute(np.array([1]))
array([1, 2])

On the other hand, each node can give you a quantum circuit, a normalization
factor, and subspaces for its input and output, which together make up a block
encoding of the vector.

TODO: Node should have a method simulate

.. rubric:: Modules

.. autosummary::
   :toctree: generated
   :recursive:

   bequem.nodes
   bequem.subspace
   bequem.circuit
   bequem.verifier

.. toctree::
   :hidden:

   references
   block_encodings
