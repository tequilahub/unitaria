import unitaria.util as util

from unitaria.subspace import Subspace, SubspaceFactor, ControlledSubspace, ZeroQubitSubspace, FullQubitSubspace
from unitaria.circuit import Circuit
from unitaria.verifier import verify

# Documentation is generated differently for these classes, see
# docs/_templates/autosummary/class.rst
from unitaria.nodes.node import Node
from unitaria.nodes.proxy_node import ProxyNode
from unitaria.nodes.multilinear_node import MultilinearNode

from unitaria.nodes.abstract_node import AbstractNode
from unitaria.nodes.block_encoding import BlockEncoding

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

from unitaria.nodes.permutation.permutation import PermuteFactors

from unitaria.nodes.fourier_transform import FourierTransform

from unitaria.nodes.nonlinear import ComponentwiseMul

from unitaria.nodes.classical.classical import Classical
from unitaria.nodes.classical.constant_integer_addition import ConstantIntegerAddition
from unitaria.nodes.classical.constant_integer_multiplication import ConstantIntegerMultiplication
from unitaria.nodes.classical.increment import Increment
from unitaria.nodes.classical.integer_addition import IntegerAddition

from unitaria.nodes.qsvt.qsvt import QSVT, QSVTCoefficients
from unitaria.nodes.inversion.pseudoinverse import Pseudoinverse
from unitaria.nodes.amplification.grover_amplification import GroverAmplification
from unitaria.nodes.amplification.linear_amplification import LinearAmplification
from unitaria.nodes.amplification.fixed_point_amplification import FixedPointAmplification

__all__ = [
    "Node",
    "Subspace",
    "Circuit",
    "verify",
    "AbstractNode",
    "Add",
    "Adjoint",
    "BlockDiagonal",
    "BlockEncoding",
    "BlockHorizontal",
    "BlockVertical",
    "Classical",
    "ComponentwiseMul",
    "ConstantIntegerAddition",
    "ConstantIntegerMultiplication",
    "ConstantUnitary",
    "ConstantVector",
    "FixedPointAmplification",
    "FourierTransform",
    "GroverAmplification",
    "Identity",
    "Increment",
    "IntegerAddition",
    "LinearAmplification",
    "Mul",
    "MultilinearNode",
    "PermuteFactors",
    "Projection",
    "ProxyNode",
    "Pseudoinverse",
    "QSVT",
    "QSVTCoefficients",
    "Scale",
    "Tensor",
    "SubspaceFactor",
    "ControlledSubspace",
    "ZeroQubitSubspace",
    "FullQubitSubspace",
    "util",
]
