## Generelle Richtlinien

* Assoziative Operationen haben trotzdem genau 2 Kinder, und keine Liste an Kindern
* Standard Projektionen sind auf 0
* Least significant bit ist das erste / oberste
* Der Baum sollte möglichst nah an der User Syntax sein
* Operationen lassen immer sowohl Matrizen als auch Vektoren zu wenn möglich, z.B A (tensor) v
* Multilineare Operationen können variable viel Argumente bekommen

## Wichtige Abstrakte Klassen

* Node
* QubitMap (bisher Projection)
* Register (bisher AtomicProjection)
* Projection (bisher custom projection) 
* Circuit
* (Vector)

## Nodes

* Multiply
* Add
* ~ConstantVector~ = ConstantMatrix
* ConstantMatrix
* TensorProduct
* Invert, Solve
* Adjoint
* BlockDiagonal
* AmplitudeAmplification
* Scale, (IncreaseNormalization)
* BlockHorizontal, (BlockVertical)
* (Convolution) (bilinear), (ConstantIntegerAddition)
* ElementwiseMultiplication (bilinear)
* (AmplitudeEstimation, PhaseEstimation)

Vielleicht von Numpy inspirieren lassen

