Documentation
=============

`unitaria` is a library for working with so called "block encodings" of matrices
and vectors. These are format for performing linear algebra calculations on
quantum computers; see also this :doc:`introduction <block_encodings>` and the
:doc:`references`, which contains a list of paper references.

The basic object of this library is a `~unitaria.Node`, which may refer to a
matrix or a vector.

>>> import unitaria as ut
>>> ut.Identity(bits=1)
Identity('subspace': Subspace(1))
>>> import numpy as np
>>> ut.ConstantVector(np.array([1, 2]))
ConstantVector('vec': array([1, 2]))

As you can see, nodes are not executed immediately. After all, the operations
should be executed on a quantum computer. Instead, you can combine nodes to form
more complex expressions, stored as a computational graph.

>>> import unitaria as ut
>>> import numpy as np
>>> print(ut.Identity(bits=1) @ ut.ConstantVector(np.array([1, 2])))
Mul
├── ConstantVector{'vec': array([1, 2])}
└── Identity{'subspace': Subspace(1)}

There are now two ways in which you can obtain the encoded vector of the above
example. Each node contains a classical implementation of the operation it
performs, available through the methods `~unitaria.Node.compute` and
`~unitaria.Node.toarray`.

>>> import unitaria as ut
>>> import numpy as np
>>> (ut.Identity(bits=1) @ ut.ConstantVector(np.array([1, 2]))).toarray().real
array([1., 2.])

On the other hand, each node can give you a quantum circuit, a normalization
factor, and subspaces for its input and output, which together make up a block
encoding of the vector. The convenience method `~unitaria.Node.simulate`
combines this data, which should produce the same result as
`~unitaria.Node.toarray`.

>>> import unitaria as ut
>>> import numpy as np
>>> (ut.Identity(bits=1) @ ut.ConstantVector(np.array([1, 2]))).simulate().real
array([1., 2.])

It can be checked wether the results of `~unitaria.Node.toarray` and
`~unitaria.Node.simulate` actually match by using `~unitaria.verifier.verify`.
This also checks other useful things, such as the number of qubits in the
circuit being correct.

Custom nodes
------------

Most basic operations are implemented already in this library. See `~unitaria.Node` for
implementing custom nodes directly, or `~unitaria.ProxyNode` for implementing custom nodes
by decomposing them into other operations.

.. only:: comment

   .. autosummary::
      :toctree: generated
      :recursive:

      unitaria


API documentation
-----------------

See the documentation of the top-level module :doc:`generated/unitaria`.

.. toctree::
   :hidden:

   references
   block_encodings
