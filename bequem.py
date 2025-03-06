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


class Projection:
    num_bits: int

registry = {}

class Mul(Node):

    def simulate(self, input: np.array | None) -> np.array:
        for child in children:
            input = child.simulate(input)
        return input


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

def execute_matrix(node: Node) -> np.array:

