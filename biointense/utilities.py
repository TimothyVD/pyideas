"""
Created on Mon Mar 25 12:04:03 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator by Tvandaele
"""
import sympy
from sympy.abc import _clash


def generate_model_from_diagram(string_list):
    '''
    Generate an ODE system based on a symbolic diagram.

    Parameters
    -----------
    string_list: list
        list containing different sublist. Each sublist contains:
        conversion of substrates and products (irreversible or reversible),
        the forward reaction rate and backward reaction rate.
        
    Returns
    --------
    system: dict
        Dictionary containing the set of ODEs which can directly fed to the
        Model class.
    
    parameters: dict
        Dictionary containing all the parameters. Each parameter is initialised
        with a value of 1, so this should be changed to the appropriate value.

    Examples
    ---------
    >>> from biointense.utilities import generate_model_from_diagram
    >>> string_list = [['S + E <=> ES', 'k1', 'k2'],
                       ['ES ==> E + P', 'kcat']]
    >>> system, parameters = generate_model_from_diagram(string_list)
    >>> print(system)
    {'dES': 'k1*S*E - k2*ES - kcat*ES',
     'dE': '- k1*S*E + k2*ES + kcat*ES',
     'dS': '- k1*S*E + k2*ES', 
     'dP': 'kcat*ES'}
    >>> print(parameters)
    {'k2': 1,
     'k1': 1,
     'kcat': 1}
     
    See also
    ---------
    biointense.Model
    '''
    system = {}
    parameters = {}
    system_list = []
    for m, i in enumerate(string_list):
        i[0] = i[0].replace(" ", "")
        if len(i) == 3 and '<=>' in i[0]:
            arrow_split = i[0].split('<=>')
            beforelist = arrow_split[0].split('+')
            afterlist = arrow_split[1].split('+')
            for n, j in enumerate(beforelist):
                if j not in system_list:
                    system_list.append(j)

                prod_of_subs = arrow_split[0].replace("+", "*")
                prod_of_prod = arrow_split[1].replace("+", "*")

                temp_string = '- {0}*{1} + {2}*{3}'.format(i[1], prod_of_subs,
                                                           i[2], prod_of_prod)

                if system.get('d' + j):
                    system['d' + j] += ' ' + temp_string
                else:
                    system['d' + j] = temp_string

            for n, j in enumerate(afterlist):
                if j not in system_list:
                    system_list.append(j)

                prod_of_subs = arrow_split[0].replace("+", "*")
                prod_of_prod = arrow_split[1].replace("+", "*")

                temp_string = '{0}*{1} - {2}*{3}'.format(i[1], prod_of_subs,
                                                         i[2], prod_of_prod)

                if system.get('d' + j):
                    system['d' + j] += ' + ' + temp_string
                else:
                    system['d' + j] = temp_string

        elif len(i) == 2 and '==>' in i[0]:
            arrow_split = i[0].split('==>')
            beforelist = arrow_split[0].split('+')
            afterlist = arrow_split[1].split('+')
            for n, j in enumerate(beforelist):
                if j not in system_list:
                    system_list.append(j)

                temp_string = '{0}*{1}'.format(i[1],
                                               arrow_split[0].replace("+",
                                                                      "*"))
                if system.get('d' + j):
                    system['d' + j] += ' - ' + temp_string
                else:
                    system['d' + j] = '-' + temp_string

            for n, j in enumerate(afterlist):
                if j not in system_list:
                    system_list.append(j)
                temp_string = '{0}*{1}'.format(i[1],
                                               arrow_split[0].replace("+",
                                                                      "*"))
                if system.get('d' + j):
                    system['d' + j] += ' + ' + temp_string
                else:
                    system['d' + j] = temp_string

        elif len(i) == 2 and '<==' in i[0]:
            arrow_split = i[0].split('<==')
            beforelist = arrow_split[0].split('+')
            afterlist = arrow_split[1].split('+')
            for n, j in enumerate(beforelist):
                if j not in system_list:
                    system_list.append(j)

                temp_string = '{0}*{1}'.format(i[1],
                                               arrow_split[0].replace("+",
                                                                      "*"))
                if system.get('d' + j):
                    system['d' + j] += ' + ' + temp_string
                else:
                    system['d' + j] = temp_string

            for n, j in enumerate(afterlist):
                if j not in system_list:
                    system_list.append(j)
                temp_string = '{0}*{1}'.format(i[1],
                                               arrow_split[0].replace("+",
                                                                      "*"))
                if system.get('d' + j):
                    system['d' + j] += ' - ' + temp_string
                else:
                    system['d' + j] = '-' + temp_string

        else:
            raise Exception(('The input {0} cannot be converted properly,'
                             'please change your input!').format(i[0]))

        for n, j in enumerate(i):
            if n > 0:
                try:
                    parameters[j]
                    raise Exception(('The parameter {0} has been defined more'
                                     'than once, please change the input!')
                                     .format(j))
                except KeyError:
                    parameters[j] = 1

    return system, parameters#, system_list


def _getCoefficients(system, variables, enzyme):
    '''Filter enzyme equations and forms out of ODE system and convert
        the filtered system to its canonical form.

    Parameters
    -----------
    system : dict
        dict containing system of equation describing the reactions.
    variables : list
        list containing all state variables
    enzyme : string
        All enzyme forms have to start with the same letters, e.g. 'En' or
        'E_'. This allows the algorithm to select the enzyme forms.

    Returns
    ---------
    coeff_matrix : sympy.Matrix
        Contains the coefficients of the canonical system of enzyme_equations.
    enzyme_forms : sympy.Matrix
        Contains all enzyme forms which are present in the system.
    enzyme_equations: sympy Matrix
        Contains the corresponding rate equation of the different enzyme
        forms.

    Notes
    ------
    The conncection between the three returns is the matrix multiplication:
    coeff_matrix*enzyme_forms = enzyme_equations

    '''

    enzyme_forms = []

    # Filter all enzyme equations and add to list
    for var in variables:
        if var.startswith(enzyme):
            enzyme_forms.append(var)

    # Set up symbolic matrix of enzyme states
    enzyme_eq_strings = [system['d'+i] for i in enzyme_forms]
    enzyme_equations = sympy.Matrix(sympy.sympify(enzyme_eq_strings,
                                                  _clash))
    # Set up symbolic matrix of enzymes
    enzyme_forms = sympy.Matrix(sympy.sympify(enzyme_forms, _clash))

    # Construct square matrix 
    square_len = len(enzyme_equations)
    #coeff_matrix = sympy.zeros(len(enzyme_equations), len(enzyme_forms))
    coeff_matrix = sympy.zeros(square_len, square_len)

    # For each enzyme equation, write the coefficient
    for i, syst in enumerate(enzyme_equations):
        for j, state in enumerate(enzyme_forms):
            coeff_matrix[i, j] = syst.coeff(state)

    return coeff_matrix, enzyme_forms, enzyme_equations


def makeQSSA(system, enzyme='E', variable='P'):
    r'''Calculate quasi steady-state equation from set of ODEs
    
    This function calculates the quasi steady-state equation for the
    variable of interest

    Parameters
    -----------
    system : dict
        dict containing system of equation describing the reactions.
    enzyme : string
        All enzyme forms have to start with the same letters, e.g. 'E' or
        'E_'. This allows the algorithm to select the enzyme forms, otherwise
        a reduction is not possible.
    variable: string
        Which rate equation has to be used to replace the enzyme forms with
        the QSSA.

    Returns
    ---------
    QSSA_var : sympy equation
        Symbolic sympy equation of variable which obeys the QSSA.

    Notes
    ------
    The idea for the calculations is based on [1]_, where the system is
    first transformed in its canonical form.

    References
    -----------
    .. [1] Ishikawa, H., Maeda, T., Hikita, H., Miyatake, K., The
        computerized derivation of rate equations for enzyme reactions on
        the basis of the pseudo-steady-state assumption and the
        rapid-equilibrium assumption (1988), Biochem J., 251, 175-181

    Examples
    ---------
    >>> from biointense.utilities import makeQSSA
    >>> system = {'dE': '-k1*E*S + k2*ES + kcat*ES',
                  'dES': 'k1*E*S - k2*ES - kcat*ES',
                  'dS': '-k1*E*S + k2*ES',
                  'dP': 'kcat*ES'}
    >>> makeQSSA(system, enzyme='E', variable='P')
    'En0*S*k1*kcat/(S*k1 + k2 + kcat)'
    '''
    # Make state var
    state_var = [i[1:] for i in system.keys() if i[0] == 'd']

    # Run _getCoefficients to get filtered rate equations
    coeff_matrix, enzyme_forms, enzyme_equations = _getCoefficients(system,
                                                                    state_var,
                                                                    enzyme)

    # Add row with ones to set the sum of all enzymes equal to En0
    coeff_matrix = coeff_matrix.col_join(sympy.ones(1, c=len(enzyme_forms)))

    # Make row matrix with zeros (QSSA!), but replace last element with
    # En0 for fixing total som of enzymes
    QSSA_matrix = sympy.zeros(coeff_matrix.shape[0], c=1)
    QSSA_matrix[-1] = sympy.sympify('En0')

    # Add column with outputs to coeff_matrix
    linear_system = coeff_matrix.row_join(QSSA_matrix)

    # Find QSSE by using linear solver (throw away one line (system is
    # closed!)) list should be passed as *args, therefore * before list
    QSSE_enz = sympy.solve_linear_system(linear_system[1:, :],
                                         *list(enzyme_forms))

    # Replace enzyme forms by its QSSE in rate equation of variable of interest
    QSSE_var = sympy.sympify(system['d' + variable], _clash)
    for enz in enzyme_forms:
        QSSE_var = QSSE_var.replace(enz, QSSE_enz[enz])

    # To simplify output expand all terms (=remove brackets) and afterwards
    # simplify the equation
    QSSE_var = sympy.simplify(sympy.expand(QSSE_var))

    return QSSE_var


def checkMassBalance(system, variables='E'):
    r"""Check mass balance of variables

    This function checks whether the sum of selected variables are equal to
    zero.

    Parameters
    -----------
    system : dict
        dict containing system of equation describing the reactions.
    variables : string
        There are two possibilies: First one can give just the first letters of
        all enzyme forms, the algorithm is selecting all variables starting
        with this combination. Second, one can give the symbolic mass balance
        him/herself the algorithm will check the mass balance. See examples!

    Returns
    ---------
    massBalance : sympy symbolics
        If this is zero then mass balance is closed, otherwise the remaining
        terms are shown.

    Examples
    ---------
    >>> from biointense.utilities import checkMassBalance
    >>> system = {'dE': '-k1*E*S + k2*ES + kcat*ES',
                  'dES': 'k1*E*S - k2*ES - kcat*ES',
                  'dS': '-k1*E*S + k2*ES',
                  'dP': 'kcat*ES'}
    >>> # Check mass balance for all variables starting with 'E'
    >>> checkMassBalance(system, variables='E')
    0
    >>> # Which is the same as the combined mass balance of E and ES
    >>> checkMassBalance(system, variables='E + ES')
    '0'
    >>> # One could also make linear combination of mass balances, this is
    >>> # especially useful for systems like NO, NO2 and N2. In which the mass
    >>> # balance for N is equal to NO + NO2 + 2*N2 = 0. Next example is to
    >>> # illustrate that mass balance is not closed for E + 2*ES.
    >>> checkMassBalance(system, variables='E + 2*ES')
    'E*S*k1 - ES*k2 - ES*kcat'
    """

    state_var = [i[1:] for i in system.keys() if i[0] == 'd']

    variables = variables.replace(" ", "")
    len_var = len(variables.split('+'))

    if len_var == 1:
        var_forms = []
        string = ''

        for var in state_var:
            if var.startswith(variables):
                var_forms.append(var)
                string = string + '+' + system['d'+var]
        massBalance = sympy.sympify(string, _clash)

    elif len_var > 1:
        var_sym = sympy.sympify(variables, _clash)
        # Set up symbolic matrix of system
        system_matrix = sympy.Matrix(sympy.sympify(system.values(), _clash))
        # Set up symbolic matrix of variables
        states_matrix = sympy.Matrix(sympy.sympify(state_var, _clash))

        massBalance = 0
        for i, var in enumerate(states_matrix):
            massBalance += var_sym.coeff(var)*system_matrix[i]

    else:
        raise Exception("The argument 'variables' needs to be provided!")

    if massBalance == 0:
        print("The mass balance is closed!")
    else:
        print(("The mass balance is NOT closed for the all var starting with '"
               "" + variables + "'! \n The following term(s) cannot be "
               "striked out: " + str(massBalance) + ""))

    return massBalance
