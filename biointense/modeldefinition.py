# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 12:02:32 2015

@author: stvhoey
"""

import re
import numpy as np
import sympy
from sympy.abc import _clash

KNOWN_NUMPY_FUN = ['exp', 'ln', 'log', 'log10', 'sqrt', 'sin', 'cos',
                   'cos', 'tan', 'abs','arcsin','arccos', 'arctan',
                   'degrees', 'radians', 'sinh', 'cosh', 'tanh',
                   'arcsinh', 'arccosh', 'arctanh', 'diff',
                   'gradient', 'exp2']

def write_whiteline(defstr):
    """
    defstr : str
        str containing the definition to solve in model
    """
    defstr += '\n'
    return defstr

def write_parameters(defstr, parameters):
    """
    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model
    parameters : dict
        key gives parameter names and values the corresponding value
    """
    for parname, parvalue in parameters.iteritems():
        defstr += "    {0} = parameters['{0}']\n".format(parname)
    return defstr

def write_independent(defstr, independent):
    """
    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model
    independent : dict
        key gives independent names
    """
    for ind_name in independent:
        defstr += "    {0} = independent['{0}']\n".format(ind_name)
    return defstr

def write_ode_indices(defstr, ode_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the ode sequence is printed

    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model
    ode_variables : list
        variable names (!sequence is important)
    """
    for i, varname in enumerate(ode_variables):
        defstr += '    {0} = odes[{1}]\n'.format(varname, str(i))
    return defstr

def write_array_extraction(defstr, ode_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the ode sequence is printed

    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model
    ode_variables : list
        variable names (!sequence is important)
    """
    if ode_variables:
        defstr += "    solved_variables = kwargs.get('ode_values')\n"

    for i, varname in enumerate(ode_variables):
        defstr += '    {0} = solved_variables[:, {1}]\n'.format(varname,
                                                                str(i))
    return defstr

def write_external_call(modelstr, externalfunctions):
    """

    Parameters
    -----------
    modelstr : str
        str containing the definition to solve in model
    varname : str
        variable name to calculate
    fname : str
        function name to call
    argnames : list
        other arguments that the function request (e.g. t or other varnames)
    """
    vardict = {}

    modelstr += "    externalfunctions = args[0]\n\n"

    for idname, iddict in externalfunctions.items():
        variable = iddict['variable']
        if vardict.get(variable) is None:
            vardict[variable] = [idname]
        else:
            vardict[variable].append(idname)

        modelstr += "    {0} = externalfunctions['{0}']['fun'](".format(idname)
        for argument in iddict['arguments']:
            modelstr += argument + ','
        modelstr += ')\n'
    modelstr += '\n'

    for variable, idnames in vardict.items():
        modelstr += '    {0} = '.format(variable)
        for i, idname in enumerate(idnames):
            if i is 0:
                modelstr += idname
            else:
                modelstr += ' + ' + idname
        modelstr += '\n'

    return modelstr

def replace_numpy_fun(m):
    """
    Replace the numpy standard functions ro np. if np. is not
    added by the user

    Parameters
    ----------
    m : str
        string to replace the numpy items from
    """
    for funt in KNOWN_NUMPY_FUN:
        m = re.sub('(?<!np\.)'+ funt, 'np.' + funt, m)
    return m

def _check_order(val_array, par_array):
    """
    """
    return np.max(par_array - val_array)

def _get_order(val_array, par_array):
    """
    """
    return np.argmax(par_array - val_array)

def _switch_order(input_val_array, input_par_array):
    """
    Switch order for two variables
    """
    val_array = input_val_array.copy(True)
    par_array = input_par_array.copy(True)
    max_diff = _get_order(val_array, par_array)

    dep_value = val_array[max_diff]
    indep_value = par_array[max_diff]


    val_array[val_array == dep_value] = -1
    par_array[par_array == dep_value] = -1

    val_array[val_array == indep_value] = dep_value
    par_array[par_array == indep_value] = dep_value

    val_array[val_array == -1] = indep_value
    par_array[par_array == -1] = indep_value

    return val_array, par_array, dep_value, indep_value

def _order_algebraic(algebraic_right_side):
    """
    Order algebraic equation to avoid referencing before assignment
    """
    # Set up symbolic matrices
    alg_key_list = algebraic_right_side.keys()
    alg_keys_matrix = sympy.Matrix(sympy.sympify(alg_key_list, _clash))
    alg_val_list = algebraic_right_side.values()
    alg_val_matrix = sympy.Matrix(sympy.sympify(alg_val_list, _clash))

    val_array, par_array = np.ma.nonzero(alg_val_matrix.jacobian(alg_keys_matrix))

    if val_array.size:
        ordered = -_check_order(val_array, par_array)
        limit = 0
        while ordered < 0 and limit < len(par_array):
            val_array, par_array, dep_val, indep_val = _switch_order(val_array,
                                                                     par_array)

            alg_key_list[dep_val], alg_key_list[indep_val] = (alg_key_list[indep_val],
                                                              alg_key_list[dep_val])

            alg_val_list[dep_val], alg_val_list[indep_val] = (alg_val_list[indep_val],
                                                              alg_val_list[dep_val])
            ordered = -_check_order(val_array, par_array)

            limit += 1

        if ordered < 0:
            raise Exception('Ordening algebraic equations failed!')

    return alg_key_list, alg_val_list

def write_algebraic_lines(defstr, algebraic_right_side):
    """
    Based on the model equations of the algebraic-part model, the equations are
    printed in the function

    Parameters
    -----------
    defstr : str
        str containing the definition to solve in model
    algebraic_right_side : dict
        dict of variables with their corresponding right hand side part of
        the equation
    """
    varnames, expressions = _order_algebraic(algebraic_right_side)
    for i, varname in enumerate(varnames):
        expression = replace_numpy_fun(expressions[i])
        #defstr += '    ' + varname + ' = ' + str(expression) + '\n'
        defstr += '    {0} = {1}\n'.format(varname, str(expression))
    return defstr

def write_algebraic_solve(defstr, algebraic_right_side, independent_var):
    """
    Based on the model equations of the algebraic-part model, the equations are
    printed in the function

    Parameters
    -----------
    defstr : str
        str containing the definition to solve in model
    algebraic_right_side : dict
        dict of variables with their corresponding right hand side part of
        the equation
    independent_var : str
        name of the independent variable
    """
    varnames, expressions = _order_algebraic(algebraic_right_side)
    for i, varname in enumerate(varnames):
        expression = replace_numpy_fun(expressions[i])
        #defstr += '    ' + varname + ' = ' + str(expression) + '\n'
        defstr += '    {0} = {1} + np.zeros(len('.format(varname, str(expression)) \
                      + independent_var + '))\n'
    return defstr

def write_ode_lines(defstr, ode_right_side):
    """
    Based on the model equations of the ode-part model, the equations are
    given in the function

    Parameters
    -----------
    defstr : str
        str containing the definition to solve in model
    algebraic_right_side : dict
        dict of variables with their corresponding right hand side part of
        the equation
    """
    for varname, expression in ode_right_side.iteritems():
        expression = replace_numpy_fun(expression)
        #defstr += '    ' + varname + ' = ' + str(expression) + '\n'
        defstr += '    d{0} = {1}\n'.format(varname, str(expression))
    return defstr

def write_derivative_return(defstr, ode_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the return sequence is returned

    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model
    ode_variables : list
        variable names (!sequence is important)
    """
    ode_variables = ["d" + variable for variable in ode_variables]
    defstr += '    return [' + ', '.join(ode_variables) + ']'
    return defstr

def write_non_derivative_return(defstr, algebraic_variables):
    """
    Based on the sequence of the variables in the variables dict,
    the return sequence is returned

    Parameters
    ----------
    defstr : str
        str containing the definition to solve in model
    ode_variables : list
        variable names (!sequence is important)
    """
    #collect and return the resulting algebraic parts
    defstr += '    nonder = np.array([' + ', '.join(algebraic_variables) + ']).T\n'
    defstr += '    return nonder'
    return defstr

def generate_ode_derivative_definition(model):
    '''Write derivative of model as definition in file

    Writes a file with a derivative definition to run the model and
    use it for other applications

    Parameters
    -----------
    model : biointense.model

    '''
    modelstr = 'def fun_ode(odes, t, parameters, *args, **kwargs):\n'
    # Get the parameter values
    modelstr = write_parameters(modelstr, model.parameters)
    modelstr = write_whiteline(modelstr)
    # Get the current variable values from the solver
    modelstr = write_ode_indices(modelstr, model._ordered_var['ode'])
    modelstr = write_whiteline(modelstr)

    # Write down external called functions - not yet provided!
    if model.externalfunctions:
        modelstr = write_external_call(modelstr, model.externalfunctions)
        modelstr = write_whiteline(modelstr)

    # Write down necessary algebraic equations (if none, nothing written)
    modelstr = write_algebraic_lines(modelstr, model.systemfunctions['algebraic'])
    modelstr = write_whiteline(modelstr)

    # Write down the current derivative values
    modelstr = write_ode_lines(modelstr, model.systemfunctions['ode'])
    modelstr = write_derivative_return(modelstr, model._ordered_var['ode'])
    return modelstr

def generate_non_derivative_part_definition(model):
    '''Write derivative of model as definition in file

    Writes a file with a derivative definition to run the model and
    use it for other applications

    Parameters
    -----------
    model : biointense.model

    '''
    modelstr = 'def fun_alg(independent, parameters, *args, **kwargs):\n'
    # Get independent
    modelstr = write_independent(modelstr, model.independent)
    modelstr = write_whiteline(modelstr)
    # Get the parameter values
    modelstr = write_parameters(modelstr, model.parameters)
    modelstr = write_whiteline(modelstr)

    # Put the variables in a separate array
    if len(model._ordered_var.get('ode', [])):
        modelstr = write_array_extraction(modelstr, model._ordered_var['ode'])
        modelstr = write_whiteline(modelstr)

    # Write down external called functions - not yet provided!
    if model.externalfunctions:
        modelstr = write_external_call(modelstr, model.externalfunctions)
        modelstr = write_whiteline(modelstr)

    # Write down the equation of algebraic
    modelstr = write_algebraic_solve(modelstr,
                                     model.systemfunctions['algebraic'],
                                     model.independent[0])
    modelstr = write_whiteline(modelstr)

    modelstr = write_non_derivative_return(modelstr, model._ordered_var['algebraic'])
    return modelstr