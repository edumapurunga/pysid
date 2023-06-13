from IPython.display import display, Math
from shutil import which

def coef_to_str(c,prec=3):
    """Converts a float coefficient (c) into a string, with precision given by prec."""
    return "{:.{precision}g}".format(c, precision = prec)


def poly_to_str(P,prec=3):
    """
    Converts a MIMO set of polynomials (P) into an equivalent display string,
    following LaTeX syntax. P must be an array of arrays. For example, let A 
    be a 2 x 2 matrix of polynomials, then
        P = [A11, A12, A21, A22]

    Parameters
    ----------
    P : ndarray of ndarray
        Set of polynomials to be displayed.
        
    prec : integer, optional
        Decimal precision for the coefficients. Default is prec = 3.
    Returns
    -------
    
    """
    if P is None:
        raise Exception("Polynomial is equal to None.")
    label = []
    aux = ""
    power = 0
    first_elem = True
    #if P.shape == (1,1):
    #    P = P.reshape(1,)
    for row in P:
        for poly in row:
            for coef in poly:
                if coef != 0:
                    if first_elem == True:
                       # Appending the first element (without the sum sign in front)
                        if power == 0 and coef==1:
                            # For monic polynomials
                            aux = aux + str(1)
                        elif power == 0 and coef!=1:
                            aux = aux + coef_to_str(coef,prec)
                        else:
                            # For non-monic polynomials
                            aux = aux + coef_to_str(coef,prec) + "q^{" + "{0}".format(-power) + "}"
                        # Sets first element flag to False
                        first_elem = False
                    elif coef > 0:
                        # Appeding a positive element
                        aux = aux + " + " + coef_to_str(coef,prec) + "q^{" + "{0}".format(-power) + "}"
                    else:
                        # Appending a negative element
                        aux = aux + " " + coef_to_str(coef,prec) + "q^{" + "{0}".format(-power) + "}"
                power = power + 1

            # Appends current polynomial to the label string
            label.append(aux)

            # Resets variables for the next polynomial
            power = 0
            first_elem = True
            aux = ""
    return label

def matrix_to_str(matrix,prec=3):
    """
    Converts a matrix of float into an equivalent display string, following LaTeX
    bmatrix syntax.

    Parameters
    ----------
    matrix : ndarray
        Matrix to be displayed.
    prec : integer, optional
        Decimal precision for the coefficients. Default is prec = 3.
    Returns
    -------

    """
    m,n = matrix.shape
    #if which('latex') is not None:
    label = r"\begin{bmatrix}"
    for row in range(m):
        for col in range(n):
            label = label + coef_to_str(matrix[row][col],prec)
            if col != n-1:
                label = label + r"&"
        if row != m-1:
            label = label + r"\\ "

    label = label + "\end{bmatrix}"
    return label

def print_matrix(matrix,prec):
    m,n = matrix.shape
    for row in range(m):
        print("|",end="")
        for col in range(n):
            if matrix[row][col] > 0:
                print(" "+coef_to_str(matrix[row][col],prec=prec),end="\t")
            else:
                print(coef_to_str(matrix[row][col],prec=prec),end="\t" )
        print("|")
                
def print_poly(P,dim,name,prec=3):
    """
    Prints a MIMO polynomial model.

    Parameters
    ----------
    P : ndarray
        Polynomial model to be displayed.
    dim : array_like
        Number of outputs (ny) and inputs (nu) in this order, that is, [ny, nu].
    name : string
        Name displayed for the polynomial model.
    prec : integer, optional
        Decimal precision for the coefficients. Default is prec = 3.
    Returns
    -------
    
    """
    if P is None:
        raise Exception("Polynomial is equal to None.")
    s = poly_to_str(P,prec)
    rows, cols = dim[0], dim[1]
    index = 0
    #print(which('latex'))
    if which('latex') is not None:
        for row in range(rows):
            for col in range(cols):
                if rows == 1 and cols == 1:
                    # Prints SISO subcase
                    display(Math(r'' + name + '(q^{-1}) = ' + s[index]))
                else:
                    if name == "C" or name == "D":
                        poly_index = "{" + str(row+1) + "}"
                        display(Math(r'' + name + "_" + poly_index + '(q^{-1}) = ' + s[index]))
                    else:
                        # Prints general MIMO case
                        poly_index = "{" + str(row+1) + str(col+1) + "}"
                        display(Math(r'' + name + "_" + poly_index + '(q^{-1}) = ' + s[index]))
                index = index + 1
    else:
        for row in range(rows):
            for col in range(cols):
                if rows == 1 and cols == 1:
                    # Prints SISO subcase
                    print(r'' + name + '(q^{-1}) = ' + s[index])
                else:
                    if name == "C" or name == "D":
                        poly_index = str(row+1)
                        print(r'' + name + poly_index + ' (q^{-1}) = ' + s[index])
                    else:
                        # Prints general MIMO case
                        poly_index = str(row+1) + str(col+1)
                        print(r'' + name + poly_index + ' (q^{-1}) = ' + s[index])
                index = index + 1
    print("\n")


def print_model(model,prec=3,only_polynomials=False):
    """
    Prints the set of polynomials that define a given model.

    Parameters
    ----------
    model : polymodel object
        Identification model object containing a set of polynomials.
    prec : integer, optional
        Decimal precision for the coefficients. Default is prec = 3.
    Returns
    
    """
    names = []
    if model.A != None:
        names.append('A')
    if model.B != None:
        names.append('B')
    if model.C != None:
        names.append('C')
    if model.D != None:
        names.append('D')
    if model.F != None:
        names.append('F')
    # Model should be a pysid.identification.models.polymodel() object
    index = 0
    ny, nu = model.ny, model.nu
    dims = [[ny,ny],[ny,nu],[ny,1],[ny,1],[ny,nu]]
    for poly in model:
        if poly is not None:
            print_poly(poly,dims[index],names[index],prec)
            index = index + 1

    if not only_polynomials:
        if hasattr(model, 'costfunction'):
            print(f'Cost function per sample: ' + coef_to_str(model.costfunction,prec))
    
        if hasattr(model, 'P'):
            print(f'Accuracy:')
            if which('latex') is not None:
                display(Math(matrix_to_str(model.P,prec)))
            else:
                print_matrix(model.P,prec)
