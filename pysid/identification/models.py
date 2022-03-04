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

    def setcov(self, V, accuracy, ncov, R):
        self.P = accuracy
        self.ecov = ncov
        self.costfunction = V
        self.R = R

def gen_poly_string(P,dim,name):
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

def gen_model_string(m):
    """
    Generates a string for displaying the contents of a polymodel object.

    Parameters
    ----------
    m : polymodel
        Polynomial object to be displayed.
    Returns
    -------
    model_str : string
        Equivalent model string, ready for display.
    """
    
    # Generating model header
    m.name = m.name.upper()
    nu, ny = m.nu, m.ny
    model_str = ""

    if nu == 1:
        model_str = model_str + "SI"
    else:
        model_str = model_str + "MI"
        
    if ny == 1:
        model_str = model_str + "SO "
    else:
        model_str = model_str + "MO " 

    model_str = model_str + m.name + " model (" + str(nu) + " in, " + str(ny) + " out)\n\n"
    
    # Defining bitmask so as to conditionally print relevant polynomials
    names = [ "A",      "B",      "C",     "D",     "F"   ]
    dims  = [ [ny,ny], [ny,nu], [ny,1],  [ny,1],  [ny,nu] ]
    mask  = [ False,   False,   False,   False,   False   ]

    if m.name == "AR":
        mask[0] = True
    elif m.name == "MA":
        mask[2] = True
    elif m.name == "ARMA":
        mask[0] = mask[2] = True
    elif m.name == "ARX":
        mask[0] = mask[1] = True
    elif m.name == "ARMAX":
        mask[0] = mask[1] = mask[2] = True
    elif m.name == "FIR":
        mask[1] = True
    elif m.name == "OE":
        mask[1] = mask[4] = True
    elif m.name == "BJ":
        mask[1] = mask[2] = mask[3] = mask[4] = True
    elif m.name == "PEM":
        mask[0] = mask[1] = mask[2] = mask[3] = mask[4] = True
    else:
        raise ValueError('Invalid model name.')

    index = 0
    for poly in m:
        if mask[index]:
            model_str = model_str + gen_poly_string(poly,dims[index],names[index]) + "\n"
        index = index + 1

    model_str = model_str + "\nMODEL PROPERTIES:"
    model_str = model_str + "\nTime delay: " + str(m.d)
    model_str = model_str + "\nSample time: " + str(m.ts)
    model_str = model_str + "\necov: " + str(m.ecov)
    model_str = model_str + "\nCost function: " + str(m.costfunction)
    model_str = model_str + "\n\nAccuracy:\n" + str(m.P)

    return model_str