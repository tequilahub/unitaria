Introduction to block encodings
===============================

	This site explains the concept of block encodings. For further reading, see
	:doc:`references`.

Block encodings allow to encode arbitrary linear operations in quantum circuits,
and thus enable an up to exponential speedup in the computation of these
operations. The speedup comes from how vectors are stored on quantum computers.
Specifically, note how for classical computer with :math:`n` bits of storage
its state can be represented by an integer between :math:`0`, corresponding
to all bits being zero, and :math:`2^n - 1`, corresponding to all bits being
one. In the context of quantum computing we would write these states as
:math:`|0\rangle` to :math:`|2^n-1\rangle`. The state of a quantum computer,
however, can also be a superposition of these classical states, which has the
form

.. math::

   |\phi\rangle = \sum_{j=0}^{2^n-1} \phi_j |j\rangle.

In what we call *amplitude encoding* this state represents the vector

.. math::

   (\phi_0, \dots, \phi_{2^n - 1}).
