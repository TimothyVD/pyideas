# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:57:22 2015

@author: timothy
"""
import numpy as np
import sympy
from sympy.abc import _clash

from biointense.modeldefinition import *


def generate_ode_sens(odefunctions, algebraicfunctions, parameters):
    '''Analytic derivation of the local sensitivities of ODEs

    Sympy based implementation to get the analytic derivation of the
    ODE sensitivities. Algebraic variables in the ODE equations are replaced
    by its equations to perform the analytical derivation.
    '''

    # Set up symbolic matrix of system states
    system_matrix = sympy.Matrix(sympy.sympify(odefunctions.values(), _clash))
    # Set up symbolic matrix of variables
    states_matrix = sympy.Matrix(sympy.sympify(odefunctions.keys(), _clash))
    # Set up symbolic matrix of parameters
    parameter_matrix = sympy.Matrix(sympy.sympify(parameters.keys(), _clash))

    # Replace algebraic stuff in system_matrix to perform LSA
    if bool(algebraicfunctions):
        # Set up symbolic matrix of algebraic
        algebraic_matrix = sympy.Matrix(sympy.sympify(
            algebraicfunctions.keys(), _clash))
        # Replace algebraic variables by its equations
        h = 0
        while (np.sum(np.abs(system_matrix.jacobian(algebraic_matrix))) != 0) and \
            (h <= len(algebraicfunctions.keys())):
            for i, alg in enumerate(algebraicfunctions.keys()):
                system_matrix = system_matrix.replace(
                                    sympy.sympify(alg, _clash),
                                    sympy.sympify(algebraicfunctions.values()[i],
                                                  _clash))
            h += 1

    # Initialize and calculate matrices for analytic sensitivity calculation
    # dfdtheta
    dfdtheta = system_matrix.jacobian(parameter_matrix)
    dfdtheta = np.array(dfdtheta)
    # dfdx
    dfdx = system_matrix.jacobian(states_matrix)
    dfdx = np.array(dfdx)
    # dxdtheta
    dxdtheta = np.zeros([len(states_matrix), len(parameters)])
    dxdtheta = np.asmatrix(dxdtheta)

    return dfdtheta, dfdx, dxdtheta

def generate_alg_sens(odefunctions, algebraicfunctions, parameters):
    """
    """
    # Set up symbolic matrix of variables
    if odefunctions:
        states_matrix = sympy.Matrix(sympy.sympify(odefunctions.keys(),
                                                   _clash))
    # Set up symbolic matrix of parameters
    parameter_matrix = sympy.Matrix(sympy.sympify(parameters.keys(), _clash))

    algebraic_matrix = _alg_swap(algebraicfunctions)

    # Initialize and calculate matrices for analytic sensitivity calculation
    # dgdtheta
    dgdtheta = algebraic_matrix.jacobian(parameter_matrix)
    dgdtheta = np.array(dgdtheta)
    # dgdx
    dgdx = None
    if odefunctions:
        dgdx = np.array(algebraic_matrix.jacobian(states_matrix))

    return dgdtheta, dgdx

def _alg_swap(algebraicfunctions):
    '''Algebraic swapping and replacing function

    This function is a helper function for _alg_LSA, the aim of this function
    is to replace algebraic variables in other algebraic equations by equations
    which are only dependent on time, parameters and ODEs.

    See also
    ---------
    _alg_LSA
    '''

    h = 0
    algebraic_matrix = sympy.Matrix(sympy.sympify(algebraicfunctions.values(),
                                                  _clash))
    algebraic_keys = sympy.Matrix(sympy.sympify(algebraicfunctions.keys(),
                                                _clash))
    while (np.sum(np.abs(algebraic_matrix.jacobian(algebraic_keys))) != 0) and (h <= len(algebraicfunctions.keys())):
        for i, alg in enumerate(algebraic_keys):
            algebraic_matrix = algebraic_matrix.replace(sympy.sympify(alg, _clash),
                                                        sympy.sympify(algebraicfunctions.values()[i], _clash))
        h += 1

    algebraic_swap = algebraic_matrix

    return algebraic_swap

def generate_ode_derivative_definition(model, dfdtheta, dfdx):
    '''Write derivative of model as definition in file

    Writes a file with a derivative definition to run the model and
    use it for other applications

    Parameters
    -----------
    model : biointense.model

    '''
    modelstr = 'def fun_ode_lsa(odes, t, parameters, *args, **kwargs):\n'
    # Get the parameter values
    modelstr = moddef.write_parameters(modelstr, model.parameters)
    modelstr = moddef.write_whiteline(modelstr)
    # Get the current variable values from the solver
    modelstr = moddef.write_ode_indices(modelstr, model._ordered_var['ode'])
    modelstr = moddef.write_whiteline(modelstr)
    # Write down necessary algebraic equations (if none, nothing written)
    modelstr = moddef.write_algebraic_lines(modelstr,
                                            model.systemfunctions['algebraic'])
    modelstr = moddef.write_whiteline(modelstr)

    # Write down external called functions - not yet provided!
    #write_external_call(defstr, varname, fname, argnames)
    #write_whiteline(modelstr)

    # Write down the current derivative values
    modelstr = moddef.write_ode_lines(modelstr, model.systemfunctions['ode'])
    modelstr = moddef.write_derivative_return(modelstr,
                                              model._ordered_var['ode'])

    modelstr += '\n    #Sensitivities\n\n'

    # Calculate number of states by using inputs
    modelstr += '    state_len = len(odes)/(len(parameters)+1)\n'
    # Reshape ODES input to array with right dimensions in order to perform
    # matrix multiplication
    modelstr += ('    dxdtheta = array(odes[state_len:].reshape(state_len, '
                 'len(parameters)))\n\n')

    # Write dfdtheta as symbolic array
    modelstr += '    dfdtheta = '
    modelstr += pprint.pformat(dfdtheta)
    # Write dfdx as symbolic array
    modelstr += '\n    dfdx = '
    modelstr += pprint.pformat(dfdx)
    # Calculate derivative in order to integrate this
    modelstr += '\n    dxdtheta = dfdtheta + dot(dfdx, dxdtheta)\n'

    modelstr += ('    return ' + str(odefunctions).replace("'", "") + " "
                 '+ list(dxdtheta.reshape(-1,))\n\n\n')

    return replace_numpy_fun(modelstr)

def generate_non_derivative_part_definition(model, dgdtheta, dgdx):
    '''Write derivative of model as definition in file

    Writes a file with a derivative definition to run the model and
    use it for other applications

    Parameters
    -----------
    model : biointense.model

    '''
    modelstr = 'def fun_alg_lsa(independent, parameters, *args, **kwargs):\n'
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
    #write_external_call(defstr, varname, fname, argnames)
    #write_whiteline(modelstr)

    # Write down the equation of algebraic
    modelstr = write_algebraic_solve(modelstr,
                                     model.systemfunctions['algebraic'],
                                     model.independent[0])
    modelstr = write_whiteline(modelstr)

    # TODO!
    modelstr += '\n    #Sensitivities\n\n'
    modelstr += '    indep_len = len(' + model.independent[0] + ')\n\n'

    # Write dgdtheta as symbolic array
    modelstr += ('    dgdtheta = np.zeros([indep_len, '
                 '' + str(len(model.systemfunctions['algebraic'].keys())) + ', '
                 '' + str(len(model.parameters.keys())) + '])\n')
    for i, alg in enumerate(model.systemfunctions['algebraic'].keys()):
        for j, par in enumerate(model.parameters.keys()):
            modelstr += ('    dgdtheta[:,' + str(i) + ',' + str(j) + '] = '
                         '' + str(dgdtheta[i, j]) + '\n')

    # Write dgdx as symbolic array
    if model.systemfunctions.get('ode', None):
        modelstr += ('    dgdx = np.zeros([indep_len, '
                     '' + str(len(model.systemfunctions['algebraic'].keys())) + ', '
                     '' + str(len(model.systemfunctions['ode'].keys())) + '])\n')
        for i, alg in enumerate(model.systemfunctions['algebraic'].keys()):
            for j, par in enumerate(model.systemfunctions['ode'].keys()):
                modelstr += ('    dgdx[:,' + str(i) + ',' + str(j) + '] = '
                             '' + str(dgdx[i, j]) + '\n')

        # The two time-dependent 2D matrices should be multiplied with each other
        # (dot product). In order to yield a time-dependent 2D matrix, this is
        # possible using the einsum function.
        modelstr += ("\n    dgdxdxdtheta = np.einsum('ijk,ikl->ijl', dgdx, "
                     "dxdtheta)\n")

        modelstr += '\n    dydtheta = dgdtheta + dgdxdxdtheta\n'

    else:
        modelstr += '\n    dydtheta = dgdtheta\n'

    modelstr += '\n    return dydtheta\n'

    return replace_numpy_fun(modelstr)
