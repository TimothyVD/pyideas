# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:57:22 2015

@author: timothy
"""
import numpy as np
import sympy
from sympy.abc import _clash

from biointense.modeldefinition import *

import pprint


def generate_ode_sens(odevar, odefunctions, algvar, algfunctions, parameters):
    '''Analytic derivation of the local sensitivities of ODEs

    Sympy based implementation to get the analytic derivation of the
    ODE sensitivities. Algebraic variables in the ODE equations are replaced
    by its equations to perform the analytical derivation.
    '''

    # Set up symbolic matrix of system states
    system_matrix = sympy.Matrix(sympy.sympify(odefunctions, _clash))
    # Set up symbolic matrix of variables
    states_matrix = sympy.Matrix(sympy.sympify(odevar, _clash))
    # Set up symbolic matrix of parameters
    parameter_matrix = sympy.Matrix(sympy.sympify(parameters, _clash))

    # Replace algebraic stuff in system_matrix to perform LSA
    if bool(algfunctions):
        # Set up symbolic matrix of algebraic
        algebraic_matrix = sympy.Matrix(sympy.sympify(algvar, _clash))
        # Replace algebraic variables by its equations
        h = 0
        while (np.sum(np.abs(system_matrix.jacobian(algebraic_matrix))) != 0) and \
            (h <= len(algvar)):
            for i, alg in enumerate(algvar):
                system_matrix = system_matrix.replace(
                                    sympy.sympify(alg, _clash),
                                    sympy.sympify(algfunctions[i],
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

def generate_alg_sens(odevar, odefunctions, algvar, algfunctions, parameters):
    """
    """
    # Set up symbolic matrix of variables
    if odefunctions:
        states_matrix = sympy.Matrix(sympy.sympify(odevar, _clash))
    # Set up symbolic matrix of parameters
    parameter_matrix = sympy.Matrix(sympy.sympify(parameters, _clash))

    algebraic_matrix = _alg_swap(algvar, algfunctions)

    # Initialize and calculate matrices for analytic sensitivity calculation
    # dgdtheta
    dgdtheta = algebraic_matrix.jacobian(parameter_matrix)
    dgdtheta = np.array(dgdtheta)
    # dgdx
    dgdx = None
    if odefunctions:
        dgdx = np.array(algebraic_matrix.jacobian(states_matrix))

    return dgdtheta, dgdx

def _alg_swap(algvar, algfunctions):
    '''Algebraic swapping and replacing function

    This function is a helper function for _alg_LSA, the aim of this function
    is to replace algebraic variables in other algebraic equations by equations
    which are only dependent on time, parameters and ODEs.

    See also
    ---------
    _alg_LSA
    '''

    h = 0
    algebraic_matrix = sympy.Matrix(sympy.sympify(algfunctions, _clash))
    algebraic_keys = sympy.Matrix(sympy.sympify(algvar, _clash))

    check_alg_der = np.sum(np.abs(algebraic_matrix.jacobian(algebraic_keys)))
    max_der_len = len(algvar)
    while (check_alg_der != 0) and (h <= max_der_len):
        for i, alg in enumerate(algebraic_keys):
            alg_sym = sympy.sympify(alg, _clash)
            alg_fun_sym = sympy.sympify(algfunctions[i], _clash)
            # Replace var by the corresponding function
            algebraic_matrix = algebraic_matrix.replace(alg_sym, alg_fun_sym)

        # Check whether any relation exist between different variables
        check_alg_der = np.sum(np.abs(algebraic_matrix.jacobian(algebraic_keys)))
        # Next round
        h += 1

    algebraic_swap = algebraic_matrix

    return algebraic_swap

def generate_ode_derivative_definition(model, dfdtheta, dfdx, parameters):
    '''Write derivative of model as definition in file

    Writes a file with a derivative definition to run the model and
    use it for other applications

    Parameters
    -----------
    model : biointense.model

    '''
    modelstr = 'def fun_ode_lsa(odes, t, parameters, *args, **kwargs):\n'
    # Get the parameter values
    modelstr = write_parameters(modelstr, model.parameters)
    modelstr = write_whiteline(modelstr)
    # Get the current variable values from the solver
    modelstr = write_ode_indices(modelstr, model._ordered_var['ode'])
    modelstr = write_whiteline(modelstr)
    # Write down necessary algebraic equations (if none, nothing written)
    modelstr = write_algebraic_lines(modelstr,
                                     model.systemfunctions['algebraic'])
    modelstr = write_whiteline(modelstr)

    # Write down the current derivative values
    modelstr = write_ode_lines(modelstr, model.systemfunctions['ode'])

    modelstr += '\n    #Sensitivities\n\n'

    # Calculate number of states by using inputs
    modelstr += '    state_len = ' + str(len(model._ordered_var['ode'])) + '\n'
    # Reshape ODES input to array with right dimensions in order to perform
    # matrix multiplication
    modelstr += ('    dxdtheta = np.array(odes[state_len:].reshape(state_len, '
                 '' + str(len(parameters)) + '))\n\n')

    # Write dfdtheta as symbolic array
    modelstr += '    dfdtheta = np.'
    modelstr += pprint.pformat(dfdtheta)
    # Write dfdx as symbolic array
    modelstr += '\n    dfdx = np.'
    modelstr += pprint.pformat(dfdx)
    # Calculate derivative in order to integrate this
    modelstr += '\n    dxdtheta = dfdtheta + np.dot(dfdx, dxdtheta)\n'

    modelstr = write_derivative_return(modelstr, model._ordered_var['ode'])
    modelstr += ' + list(dxdtheta.reshape(-1,))\n\n'

    return replace_numpy_fun(modelstr)

def generate_non_derivative_part_definition(model, dgdtheta, dgdx, parameters):
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

        modelstr += "\n    dxdtheta = kwargs.get('dxdtheta')\n"

    # Write down the equation of algebraic
    modelstr = write_algebraic_solve(modelstr,
                                     model.systemfunctions['algebraic'],
                                     model._independent_names[0])
    modelstr = write_whiteline(modelstr)

    # TODO!
    modelstr += '\n    #Sensitivities\n\n'
    modelstr += '    indep_len = len(' + model._independent_names[0] + ')\n\n'

    # Write dgdtheta as symbolic array
    modelstr += ('    dgdtheta = np.zeros([indep_len, '
                 '' + str(len(model.systemfunctions['algebraic'].keys())) + ', '
                 '' + str(len(parameters)) + '])\n')
    for i, alg in enumerate(model.systemfunctions['algebraic'].keys()):
        for j, par in enumerate(parameters):
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
