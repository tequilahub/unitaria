from abc import abstractmethod

import numpy as np
from rich.panel import Panel

from bequem.nodes.node import Node
from bequem.circuit import Circuit
from bequem.qubit_map import Subspace

class ProxyNode(Node):
    """
    Abstract class for nodes that are defined in terms of other nodes

    This is useful when the given definition is used in the construction of
    the circuit or QubitMaps, but a simpler representation is avilable for
    :py:func:`compute`. Additionally, the simpler structure is used for printing
    and serialization.

    For an example of how to use this class see :py:class:`Add` and
    :py:class:`Mul`.
    """

    _definition: Node | None = None

    @abstractmethod
    def definition(self) -> Node:
        """
        The definition of this node, which is used to implement all other
        abstract methods of :py:class:`Node`. The other methods can be
        overwritten to give a more efficient implementation.
        """
        raise NotImplementedError

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        # TODO: Use something more concise for lazy calculation, instead of manually
        #  checking for None everywhere
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.compute(input)

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.compute_adjoint(input)

    def circuit(self) -> Circuit:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.circuit()

    def subspace_in(self) -> Subspace:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.subspace_in()

    def subspace_out(self) -> Subspace:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.subspace_out()

    def normalization(self) -> float:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.normalization()

    def phase(self) -> float:
        if self._definition is None:
            self._definition = self.definition()
        return self._definition.phase()

    def tree_label(self, verbose: bool = False):
        label = super().tree_label()
        if not verbose:
            return label
        else:
            if self._definition is None:
                self._definition = self.definition()
            return Panel(self._definition.tree(verbose=True, holes=self.children()), title=label, title_align="left")
