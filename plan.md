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

1. [x] Caching
1. [ ] Check dass case_one.total_qubits = case_zero.total_qubits
1. [ ] Alles in Datein aufspalten
1. [ ] Dependencies qulacs, cirq, qiskit
1. [ ] Global phase in tequila implementieren (Ram)
2. [ ] Identity Gate (Ram)
3. [ ] Circuit für QubitMap (Oliver)
3. [x] phase (Oliver)
4. [x] Componentwise Multiplication (Matthias)
5. [x] Circuit Synthesis (Matthias)
6. [ ] Nonlinear Beispiel (Jessica)
7. [ ] FEM Beispiel (Matthias)
8. [ ] QSVT (Matthias)
9. [ ] Laplace / Konvolution testen <-- für nächstes Treffen
10. [ ] Ancillas (Oliver)
10. [ ] Measurements
11. [ ] CI
12. [ ] Documentation
13. [ ] plan.md nicht auf PyPi hochladen
14. [ ] Verify mit zufälliger Basis
15. [ ] Better custom verificaitions

## Offene Fragen

* Wie können wir mit Global Phase umgehen? Tequila kann das im Prinzip nicht
  (Benötigt für `Add`, `ConstantVector`)
* Soll normalization ein float oder complex sein?
