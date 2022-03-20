"""
    This module implements the models used in the pysid package
"""
#%%
# Imports

# Classes
class polymodel():
    """
    This model represents a general linear polynomial model as follows
        A(q) y(t) = B(q) / F(q) u(t) + C(q)/ D(q) e(t) 
    """

    # Initialization
    def __init__(self, name, A, B, C, D, F, d, data, nu, ny, ts):
        self.name = name
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.F = F
        self.d = d
        self.data = data
        self.nu = nu
        self.ny = ny
        self.ts = ts

        # TODO: Ensure that setcov() is always called in pemethod.py
        self.ecov = -1

    # Iterable
    def __iter__(self):
        return (i for i in (self.A, self.B, self.C, self.D, self.F))

    def __repr__(self):
        polymodelname = type(self).__name__
        s = '{}({!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r})'\
            .format(polymodelname, self.name, self.A, self.B, self.C, self.D, self.F, self.d, 'data', self.nu, self.ny, self.ts)
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

    def gen_poly_string(self, P, dim, name):
        """
        Generates a string for displaying a MIMO polynomial model.

        Parameters
        ----------
        P : ndarray
            Polynomial to be converted.
        dim : array_like
            Number of outputs (ny) and inputs (nu) in this order, that is, [ny, nu].
        name : string
            Name displayed for the polynomial model.
        Returns
        -------
        poly_str : string
            Equivalent polynomial string, ready for display.

        """

        poly_str = ""
        rows, cols = dim[0], dim[1]
        for row in range(rows):
            for col in range(cols):
                if rows == 1 and cols == 1:
                    # SISO subcase
                    poly_str = poly_str + name + " = " + str(P)
                else:
                    # General MIMO case
                    poly_str = poly_str + name + str(row+1) + str(col+1) + " = " + str(P[row][col]) + "\n"

        return poly_str

    def gen_model_string(self):
        """
        Generates a string for displaying the contents of a polymodel object.

        Parameters
        ----------

        Returns
        -------
        model_str : string
            Equivalent model string, ready for display.
        """

        # Generating model header
        self.name = self.name.upper()
        nu, ny = self.nu, self.ny
        model_str = ""

        if nu == 1:
            model_str = model_str + "SI"
        else:
            model_str = model_str + "MI"

        if ny == 1:
            model_str = model_str + "SO "
        else:
            model_str = model_str + "MO "

        model_str = model_str + self.name + " model (" + str(nu) + " in, " + str(ny) + " out)\n\n"
        
        # Defining bitmask so as to conditionally print relevant polynomials
        names = [ "A",      "B",      "C",     "D",     "F"   ]
        dims  = [ [ny,ny], [ny,nu], [ny,1],  [ny,1],  [ny,nu] ]
        mask  = [ False,   False,   False,   False,   False   ]

        if self.name == "AR":
            mask[0] = True
        elif self.name == "MA":
            mask[2] = True
        elif self.name == "ARMA":
            mask[0] = mask[2] = True
        elif self.name == "ARX":
            mask[0] = mask[1] = True
        elif self.name == "ARMAX":
            mask[0] = mask[1] = mask[2] = True
        elif self.name == "FIR":
            mask[1] = True
        elif self.name == "OE":
            mask[1] = mask[4] = True
        elif self.name == "BJ":
            mask[1] = mask[2] = mask[3] = mask[4] = True
        elif self.name == "PEM":
            mask[0] = mask[1] = mask[2] = mask[3] = mask[4] = True
        else:
            raise ValueError('Invalid model name.')

        index = 0
        for poly in self:
            if mask[index]:
                model_str = model_str + self.gen_poly_string(poly,dims[index],names[index]) + "\n"
            index = index + 1

        model_str = model_str + "\nMODEL PROPERTIES:"
        model_str = model_str + "\nSample time: " + str(self.ts)
        model_str = model_str + "\nTime delay:\n" + str(self.d)

        # TODO: Define setcov() parameters for all pemethod.py functions
        if self.ecov != -1:
            model_str = model_str + "\n\necov: " + str(self.ecov)
            model_str = model_str + "\nCost function per sample: " + str(self.costfunction)
            model_str = model_str + "\n\nAccuracy:\n" + str(self.P)

        model_str = model_str + "\n________________________________________________________\n"

        return model_str