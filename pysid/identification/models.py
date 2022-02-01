"""
    This module implements the models used in the pysid package
"""
# Imports
from ..io.print import poly_to_str

# Classes
class polymodel():
    """
        This model represents a general linear polynomial model as follows
        A(q) y(t) = B(q) / F(q) u(t) + C(q)/ D(q) e(t) 
    """

    # Initialization
    def __init__(self, name, A, B, C, D, F, d, data, ni, no, ts):
        self.name = name
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.F = F
        self.d = d
        self.data = data
        self.ni = ni
        self.no = no
        self.ts = ts

    # Iterable
    def __iter__(self):
        return (i for i in (self.A, self.B, self.C, self.D, self.F))

    def __repr__(self):
        polymodelname = type(self).__name__
        s = '{}({!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r})'\
            .format(polymodelname, self.name, self.A, self.B, self.C, self.D, self.F, self.d, 'data', self.ni, self.no, self.ts)
        return s

    def __str__(self):
        s = 'A ' + self.name + ' model'
        polymodelname = type(self).__name__
        return s

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def getdata(self):
        return self.data

    def setcov(self, V, accuracy, ncov):
        self.P = accuracy
        self.ecov = ncov
        self.costfunction = V
