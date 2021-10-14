"""
    This module implements the models used in the pysid package
"""
# Imports

# Classes
class polymodel():
    """
        This model represents a general linear polynomial model as follows
        A(q) y(t) = B(q) / F(q) u(t) + C(q)/ D(q) e(t) 
    """

    # Initialization
    def __init__(self, A, B, C, D, F):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.F = F

    # Iterable
    def __iter__(self):
        return (i for i in (self.A, self.B, self.C, self.D, self.F)

    def __str__(self):
        return str(tuple(self.A, self.B, self.C, self.D, self.F))

    def __eq__(self, other):
        return tuple(self) == tuple(other)



