# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 17:32:29 2015

@author: timothy
"""
from __future__ import division

import numpy as np

from biointense.modelbase import BaseModel
from biointense.solver import HybridOdeSolver, \
    HybridOdeintSolver, HybridOdespySolver


def michaelis_menten():
    def fun_ODE(ODES, t, Parameters):
        Ks = Parameters['Ks']
        Vmax = Parameters['Vmax']

        P = ODES[0]
        S = ODES[1]

        v = S*Vmax/(Ks + S)

        dP = v
        dS = -v
        return [dP, dS]

    def fun_alg(t, Parameters, ODES):
        Ks = Parameters['Ks']
        Vmax = Parameters['Vmax']

        P = ODES[:,0]
        S = ODES[:,1]

        v = S*Vmax/(Ks + S) + np.zeros(len(t))

        algebraic = np.array([v]).T

        return algebraic

    system = {'v' : 'Vmax*S/(Ks + S)',
              'dS': '-v',
              'dP' : 'v'}
    parameters = {'Vmax': 1e-1, 'Ks': 0.5}

    model = BaseModel(system, 'test', parameters)
    model.systemfunctions['algebraic'] = fun_alg
    model.systemfunctions['ode'] = fun_ODE

    model.parameters = {'Vmax': 1e-1, 'Ks': 0.5}
    model.initial_conditions = {'S': 0.5, 'P': 0.0}

    model.independent_values = np.linspace(0, 72, 1000)
    model.variables = {'algebraic': ['v'], 'ode': ['P', 'S']}

    solver1 = HybridOdeSolver(model)
    result1 = solver1.solve()

    solver2 = HybridOdeintSolver(model)
    result2 = solver2.solve()

    solver3 = HybridOdespySolver(model)
    result3 = solver3.solve()

    return result1, result2, result3

if __name__ == "__main__":
    michaelis_menten()
