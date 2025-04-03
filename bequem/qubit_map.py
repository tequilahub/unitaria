@dataclass
class QubitMap:
    components: list[Register]

    def simplify(self) -> QubitMap:
        simplified = []
    
        for register in self.registers:
            if type(register) is Case:
                case_zero = register.case_zero.simplify()
                case_one = register.case_one.simplify()
                # TODO: Allow for more lenient matching, e.g. AncillaBit == IdBit
                if case_zero == case_one:
                    simplified.append(case_zero)
                    simplified.append(IdBit)
                else:
                    simplified.append(Case(case_zero, case_one))
            else:
                simplified.append(register)

        return QubitMap(simplified)
            

    def reduce(self) -> QubitMap:
        return QubitMap([
            register for register in self.registers
            if type(register) is IdBit or type(register) is Case or type(register) is Projection
        ])

    def is_all_zeros(self) -> bool:
        return all([register is ZeroBit for register in self.registers])

    def test_basis(self, bits: int):
        raise NotImplementedError

    def enumerate_basis() -> list[int]:
        raise NotImplementedError

    def project(self, vector: np.array) -> np.array:
        return vector[self.enumerate_basis()]

Register = ZeroBit | IdBit | AncillaBit | BorrowedAncillaBit | Case | Projection

@dataclass
class ZeroBit:
    pass

@dataclass
class IdBit:
    pass

@dataclass
class AncillaBit:
    pass

@dataclass
class BorrowedAncillaBit:
    pass

@dataclass
class Case:
    case_zero: QubitMap
    case_one: QubitMap

@dataclass
class Projection:
    circuit: Circuit

