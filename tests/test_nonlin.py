import pytest

import numpy as np
from bequem.nodes.constant import ConstantVector

from bequem.nodes.integer_arithmetic import Increment
from bequem.nodes import Add, Mul, ConstantUnitary, Scale
from bequem.nodes.nonlinear import ComponentwiseMul 


# v0 = [1,1] 
# g: [x1, x2]  -->  [1,1] - 1/8 [(x1+x2)^2, (x1-x2)^2]

# implements g^n(v0)


v0 = ConstantVector(np.array([1, 1]))


@pytest.mark.parametrize("n", range(1, 3))
def test_nonlin(n:int): 
    v0 = ConstantVector(np.array([1, 1]))

    A = ConstantUnitary((1/np.sqrt(2)) *np.array([[1,1],
                                          [1,-1]]))
    for i in range(0,n): 
        v0 = Mul(v0, A)
                
        v0 = Add(ConstantVector(np.array([1, 1])),
                Scale(ComponentwiseMul(v0.subspace_out)@(v0&v0),
                     -1/8))

    v0.verify()    










