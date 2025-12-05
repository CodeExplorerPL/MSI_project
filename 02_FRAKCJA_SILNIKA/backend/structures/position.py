""" Klasa pozycji """


class Position:
    """Reprezentuje pozycję X, Y na mapie."""
    x: float
    y: float

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getstate__(self):
        return self.x, self.y

    def __setstate__(self, state):
        self.x, self.y = state

