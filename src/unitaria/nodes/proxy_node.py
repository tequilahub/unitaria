from abc import abstractmethod
from typing import Sequence

import numpy as np
from rich.panel import Panel

from unitaria.nodes.node import Node
from unitaria.circuit import Circuit
from unitaria.subspace import Subspace


class ProxyNode(Node):
    """
    Abstract class for nodes that are defined in terms of other nodes

    This is useful when the given definition is used in the construction of
    the circuit or QubitMaps, but a simpler representation is avilable for
    `compute`. Additionally, the simpler structure is used for printing and
    serialization.

    For an example of how to use this class see `Add` and `Mul`.
    """

    _definition: Node | None = None

    @abstractmethod
    def definition(self) -> Node:
        """
        The definition of this node.

        This is used to implement all other
        abstract methods of `Node`. The other methods can be overwritten to give
        a more efficient implementation.
        """
        raise NotImplementedError

    def _definition_internal(self) -> Node:
        """
        :no-index:
        """
        if self._definition is None:
            self._definition = self.definition()
        return self._definition

    def compute(self, input: np.ndarray | None) -> np.ndarray:
        # TODO: Use something more concise for lazy calculation, instead of manually
        #  checking for None everywhere
        definition = self._definition_internal()
        return definition.compute(input)

    def compute_adjoint(self, input: np.ndarray | None) -> np.ndarray:
        definition = self._definition_internal()
        return definition.compute_adjoint(input)

    def _subspace_in(self) -> Subspace:
        definition = self._definition_internal()
        return definition.subspace_in

    def _subspace_out(self) -> Subspace:
        definition = self._definition_internal()
        return definition.subspace_out

    def _normalization(self) -> float:
        definition = self._definition_internal()
        return definition.normalization

    def _circuit(
        self, target: Sequence[int], clean_ancillae: Sequence[int], borrowed_ancillae: Sequence[int]
    ) -> Circuit:
        definition = self._definition_internal()
        return definition.circuit(target, clean_ancillae, borrowed_ancillae)

    def clean_ancilla_count(self) -> int:
        definition = self._definition_internal()
        return definition.clean_ancilla_count()

    def borrowed_ancilla_count(self) -> int:
        definition = self._definition_internal()
        return definition.borrowed_ancilla_count()

    def tree_label(self, verbose: bool = False):
        label = super().tree_label()
        if not verbose:
            return label
        else:
            definition = self._definition_internal()
            return Panel(definition.tree(verbose=True, holes=self.children()), title=label, title_align="left")
