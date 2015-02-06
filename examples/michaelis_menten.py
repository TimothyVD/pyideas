# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 17:32:29 2015

@author: timothy
"""
from __future__ import division

import numpy as np
import pandas as pd

from biointense.modelbase import BaseModel
from biointense.model import Model
from biointense.solver import HybridOdeSolver, \
    HybridOdeintSolver, HybridOdespySolver


def michaelis_menten_old():

    from biointense import DAErunner

    ODE = {'dS': '-v',
           'dP': 'v'}
    Algebraic = {'v': 'Vmax*S/(Ks + S)'}
    parameters = {'Vmax': 1e-1, 'Ks': 0.5}

    model = DAErunner(ODE=ODE, Algebraic=Algebraic, Parameters=parameters,
                      Modelname='MichaelisMenten', print_on=False)

    model.set_initial_conditions({'dS': 0.5, 'dP': 0.0})
    model.set_xdata({'start': 0, 'end': 72, 'nsteps': 1000})
    #model.set_measured_states(['S','X'])
    #model.variables = {'algebraic': ['v'], 'ode': ['P', 'S']}

    result1 = model.solve_ode(plotit=False)
    model.solve_algebraic(plotit=False)

    return pd.concat([result1, model.algeb_solved], axis=1)


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

    system = {'v': 'Vmax*S/(Ks + S)',
              'dS': '-v',
              'dP': 'v'}
    parameters = {'Vmax': 1e-1, 'Ks': 0.5}

    #model = BaseModel('test')
    #model.systemfunctions['algebraic'] = fun_alg
    #model.systemfunctions['ode'] = fun_ODE
    model = Model('MichaelisMenten', system, parameters)
    #model.fun_alg = fun_alg
    #model.fun_ODE = fun_ODE

    #model.parameters = {'Vmax': 1e-1, 'Ks': 0.5}
    model.initial_conditions = {'S': 0.5, 'P': 0.0}

    #model.independent_values = np.linspace(0, 72, 1000)
    #model.variables = {'algebraic': ['v'], 'ode': ['P', 'S']}

    model.set_independent('t', np.linspace(0, 72, 1000))
    model.initialize_model()

    solver1 = HybridOdeSolver(model)
    result1 = solver1.solve()

    solver2 = HybridOdeintSolver(model)
    result2 = solver2.solve()

    solver3 = HybridOdespySolver(model)
    result3 = solver3.solve()

    return result1, result2, result3

if __name__ == "__main__":
    result1, result2, result3 = michaelis_menten()
    result1.plot()

    result = michaelis_menten_old()
    result.plot()
