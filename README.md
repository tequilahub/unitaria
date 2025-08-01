# unitaria :rainbow:

`unitaria` is a library for working with so called "block encodings" of matrices
and vectors. These are format for performing linear algebra calculations on
quantum computers. It allows constructing quantum algorithms using a simple,
`numpy`-like syntax.

```python
>>> from bequem.nodes import ConstantVector, Identity
>>> import numpy as np
>>> result = Identity(1) @ ConstantVector(np.array([1, 2]))
>>> result
Mul
├── ConstantVector{'vec': array([3, 4])}
└── Identity{'subspace': Subspace(1)}
>>> result.toarray().real
array([3., 4.])
>>> result.normalization
5.0
>>> result.circuit
Circuit(tq_circuit=circuit: 
Ry(target=(0,), parameter=1.8545904360032246)
GlobalPhase(target=(), control=(), parameter=0.0)
)
```

## Installation

The best way to install this library is using `pip`:

```sh
pip install unitaria
```

This installs everything needed to work with `unitaria`, including the
simulation backend `qulacs`. Additional backends compatible with `tequila`,
which is for commincating with the backends, can also be installed, see
[tequila](https://github.com/tequilahub/tequila).

## Development

To install this library locally, clone this repository and run
```sh
pip install --editable .
```

To run the test suite you can then execute
```sh
pytest
```

To build the documentation, navigate to the `/docs` folder and run
```sh
rm -r generated
make html
```
This requires some additional dependencies, which can be installed using
```sh
pip install --editable .[docs]
```
