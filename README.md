# unitaria :rainbow:

:warning: The main functionality of this project is still being developed. All interfaces and features may still change, and no compatibility is guaranteed.

`unitaria` is a library for working with so called "block encodings" of matrices
and vectors. These are format for performing linear algebra calculations on
quantum computers. It allows constructing quantum algorithms using a simple,
`numpy`-like syntax.

```python
>>> import unitaria as ut
>>> import numpy as np
>>> result = ut.Identity(bits=1) @ ut.ConstantVector(np.array([3, 4]))
>>> print(result.draw())
Mul
├── ConstantVector{'vec': array([3, 4])}
└── Identity{'subspace': Subspace(1)}
>>> result.toarray().real
array([3., 4.])
>>> result.normalization
np.float64(5.0)
>>> result.circuit()
Circuit(_tq_circuit=circuit: 
GlobalPhase(target=(), control=(), parameter=0.0)
Ry(target=(0,), parameter=1.8545904360032246)
Rx(target=(0,), parameter=0.0)
Rx(target=(0,), parameter=-0.0)
Rx(target=(0,), parameter=0.0)
)

```

[**Documentation**](https://tequilahub.github.io/unitaria/index.html)

## Installation

The best way to install this library is using `pip`:

```sh
pip install unitaria
```

This installs everything needed to work with `unitaria`, including the
simulation backend `qulacs`. Additional backends compatible with `tequila`,
which is used for communcating with the backends, can also be installed, see
[tequila](https://github.com/tequilahub/tequila).

## Python versions

`unitaria` requires at least Python version 3.12, and follows [Numpy's deprecation policy](https://numpy.org/neps/nep-0029-deprecation_policy.html),
i.e. at least Python 3.13 will be required starting April 2027.

## Development

To install this library locally, clone this repository and run
```sh
pip install --editable .
```

To run the test suite you can then execute
```sh
pytest
```

To build the documentation, some additional dependencies are required, which can be installed using
```sh
pip install --editable .[docs]
```
Then navigate to the `/docs` folder and run
```sh
rm -r generated
make html
```

If you get the error `locale.Error: unsupported locale setting`, try adding the environment variable `LC_ALL=C.UTF-8`:
```sh
LC_ALL=C.UTF-8 make html
```
