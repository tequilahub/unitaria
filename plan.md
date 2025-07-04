## Generelle Richtlinien

* Assoziative Operationen haben trotzdem genau 2 Kinder, und keine Liste an Kindern
* Standard Projektionen sind auf 0
* Least significant bit ist das erste / oberste
* Der Baum sollte möglichst nah an der User Syntax sein
* Operationen lassen immer sowohl Matrizen als auch Vektoren zu wenn möglich, z.B A (tensor) v
* Multilineare Operationen können variable viel Argumente bekommen

## Wichtige Abstrakte Klassen

* [x] Node
* [x] QubitMap (bisher Projection)
* [x] Register (bisher AtomicProjection)
* [ ] Projection (bisher custom projection) 
* [x] Circuit
* [ ] (Vector)

## Nodes

* [x] Multiply
* [x] Add
* [x] ConstantVector
* [ ] ConstantUnitary
* [ ] ConstantMatrix
* [x] TensorProduct
* [ ] Invert, Solve
* [x] Adjoint
* [ ] BlockDiagonal
* [ ] AmplitudeAmplification
* [x] Scale, (IncreaseNormalization)
* [x] BlockHorizontal, (BlockVertical)
* [x] (Convolution) (bilinear), (Increment)
* [ ] ElementwiseMultiplication (bilinear)
* [ ] (AmplitudeEstimation, PhaseEstimation)
* [ ] QFT

Vielleicht von Numpy inspirieren lassen


## TODOS

* Move dimension from subspace to node (discuss next meeting)
* Bilineare Operationen besseres interface, z.B. ComponentwiseMul (Matthias)
* Dependencies qulacs, ~cirq~, ~qiskit~ (Matthias)
* Alles in Datein aufspalten
* Nonlinear Beispiel (Jessica)
* QSVT (Matthias)
* QFT (Matthias)
* Clean Ancillas (Oliver) (Pull request)
* Dirty Ancillas (Oliver) (Pull request)
* Measurements: norm() -> float (Ram)
* CI
* Documentation
* plan.md nicht auf PyPi hochladen
* Verify mit zufälliger Basis
* Better custom verifications
* Better circuit output
* Permutations


## Example projects

* FEM 1d toeplitz + Inverse?
* Gauss convolution
* Haar transformation
* Faber matrix
