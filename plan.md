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

* Abstract operations without circuit implementation
* Operations from circuit implementations (Ram)
* Nonlinear Beispiel (Jessica)
* plan.md nicht auf PyPi hochladen
* Verify mit zufälliger Basis
* Better custom verifications
* Permutations
* Divide Identity into from dimension / from qubits
* Projection (Postselection) Node (Matthias)
* Ancilla als Eigenschaft von Circuit statt Subspace (Oliver)
* tq_circuit Zugriffe entfernen
* Qubits minimieren, z.B. Tensor sequentiell/parallel
* Soll `*` auch für Tensor operationen benutzt werden?
* Subspace.circuit should be cached property

## Example projects

* FEM 1d toeplitz + Inverse?
* Gauss convolution
* Haar transformation
* Faber matrix
