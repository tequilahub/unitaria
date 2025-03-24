class Node:

    uuid: Uuid
    children: list[Node]

    def __init__(children: list[Node]=[]):
        self.children = children

    def serialize_data(self) -> str | None:
        return None

    def projection_in(self) -> Projection:

    def projection_out(self) -> Projection:

    def is_vector(self) -> bool:
        self.projection_in() == Projection(0)

    # statt np.array vielleicht Klasse, die Tensorstruktur festhält
    def simulate(self, input: np.array | None) -> np.array:
        raise NotImplementedError

    def circuit(self) -> Circuit:
        raise NotImplementedError


class Projection:
    num_bits: int

registry = {}

class Mul(Node):

    def simulate(self, input: np.array | None) -> np.array:
        for child in children:
            input = child.simulate(input)
        return input

    def circuit(self) -> Circuit:
        circuit = Circuit()
        for child in children:
            circuit.append(child.circuit())

        return circuit


class Solve(Node):

    def __init__(self, A, b, condition):
        super().__init__(self, children=[A, b], condition)

    def __init__(self, children, data): # Muss existieren für alle Nodes
        super().__init__(self, children)
        assert len(children) == 2
        assert data is None
        assert not children[0].is_vector()
        assert children[1].is_vector()

    def simulate(self, input: np.array | None) -> np.array:
        ...

    def circuit(self) -> Circuit:
        angles = solver_angles(self.condition)
        A_circuit = A.circuit()
        b_circuit = b.circuit()

        a = AncillaBits()
        d = DataBits()
        circuit = Circuit(a, d)
        circuit.append(A_circuit, d, a)
        circuit.append(b_circuit, d, a)
        circuit.rot(angle0, a)
        circuit.append(A_circuit.inverse(), d, a)
        circuit.rot(angle1, a)
        circuit.append(A_circuit, d, a)
        circuit.append(b_circuit, d, a)
        circuit.append(b_circuit.inverse(), d, a)
        circuit.rot(angle2, a)
        circuit.append(A_circuit.inverse(), d, a)
        circuit.rot(angle3, a)


class Vec(Node):

    def __init__(self, vec):
        super().__init__(self, children=[], None)
        self.data = vec

    def __init__(self, children, data): # Muss existieren für alle Nodes
        super().__init__(self, children)
        assert len(children) == 0
        assert not children
        self.data = np.array(data)

    def simulate(self, input: np.array | None) -> np.array:
        assert input is None
        return self.data

class Convolution(Node):
    def simulate(self, input: np.array | None) -> np.array:
        kernel = self.children[0].simulate(None)
        return convolve(kernel, input)


class Add(Node):

    def __init__(self, children, data=None):
        super().__init(self, children)
        if data is not None:
            raise BequemParseError


registry["vec"] = Vec


def parse():
    try:
        registry["vec"](children=[], data="4585493898")
    catch BequemParseError:
        pass

