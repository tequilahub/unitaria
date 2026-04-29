# unitaria :rainbow:

`unitaria` is a library for working with so called "block encodings" of matrices
and vectors. These are format for performing linear algebra calculations on
quantum computers. It allows constructing quantum algorithms using a simple,
`numpy`-like syntax.

```python
>>> import unitaria as ut
>>> import numpy as np
>>> result = ut.Identity(ut.Subspace.from_dim(2)) @ ut.ConstantVector(np.array([3, 4]))
>>> print(result.draw())
Mul
├── Identity{'subspace': Subspace("#")}
└── ConstantVector{'vec': array([3, 4])}
>>> result.toarray().real
array([3., 4.])
>>> result.normalization
np.float64(5.0)
>>> result.circuit()
Circuit(_tq_circuit=circuit: 
GlobalPhase(target=(), control=(), parameter=0.0)
Ry(target=(0,), parameter=1.854590436003224)
, n_qubits=1)

```

[**Documentation**][docs]

## Getting started

The best way to install this library is using `pip`:

```sh
pip install unitaria
```

This installs everything needed to work with `unitaria`, including the
simulation backend `qulacs`. Additional backends compatible with `tequila`,
which is used for communcating with the backends, can also be installed, see
[tequila](https://github.com/tequilahub/tequila).

`unitaria` aims to be as intuitive as possible. Most operators do exactly what
you would expect them to. To construct a tridiagonal matrix, you can, e.g., write
```python
import unitaria as ut
N = 3
inc = ut.Increment(bits=N)
laplace = (2 * ut.Identity(dim=2**N) - inc - inc.adjoint())[:-1, :-1]
```

If you are not sure how to construct a matrix or vector, you can use the
`ConstantMatrix` or `ConstantVector` functions.
```python
import unitaria as ut
import numpy as np
v = ut.ConstantVector(np.array([1, 2, 3]))
A = ut.ConstantMatrix(np.array([[1, 2], [3, 4]]))
```
Note, however, that this will typically not yield efficient quantum circuits.

For a list of all implemented matrices, vectors, and operations check
out the [documentation][docs]. Additional examples are available under
[`/examples`](/examples).

## Contributing

We welcome contributions to `unitaria`. Check out the [Contributing guildlines](CONTRIBUTING.md) for details.

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
pip install --group docs --editable .
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

## Python versions

`unitaria` requires at least Python version 3.12, and follows [Numpy's deprecation policy](https://numpy.org/neps/nep-0029-deprecation_policy.html),
i.e. at least Python 3.13 will be required starting April 2027.

## Versioning

Unitaria follows [SemVer](https://semver.org/spec/v2.0.0.html) conventions.

[docs]: https://tequilahub.github.io/unitaria/git/index.html
