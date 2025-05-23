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

* [y] Multiply
* [x] Add
* [x] ~ConstantVector~ = ConstantMatrix
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

Vielleicht von Numpy inspirieren lassen


## TODOS

* Bilineare Operationen besseres interface, z.B. ComponentwiseMul (Matthias)
* Qulacs global phase (Oliver)
* Dependencies qulacs, cirq, qiskit (Matthias)
* Alles in Datein aufspalten
* Global phase in tequila implementieren (Ram)
* Nonlinear Beispiel (Jessica)
* FEM Beispiel (Matthias)
* QSVT (Matthias)
* Clean Ancillas (Oliver) (Pull request)
* Dirty Ancillas (Oliver)
* Measurements: norm() -> float (Ram)
* CI
* Documentation
* plan.md nicht auf PyPi hochladen
* Verify mit zufälliger Basis
* Better custom verificaitions

## Offene Fragen

* Wie können wir mit Global Phase umgehen? Tequila kann das im Prinzip nicht
  (Benötigt für `Add`, `ConstantVector`)
* Soll normalization ein float oder complex sein?
