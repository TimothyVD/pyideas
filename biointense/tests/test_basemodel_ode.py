# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:03:19 2015

@author: joris
"""
from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal

from biointense.modelbase import BaseModel
from biointense.solver import OdeSolver


def ode_model_function(ODES, t, parameters):
    K_S = parameters['K_S']
    Q_in = parameters['Q_in']
    S_in = parameters['S_in']
    V = parameters['V']
    Ys = parameters['Ys']
    mu_max = parameters['mu_max']

    S = ODES[0]
    X = ODES[1]

    dS = Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X
    dX = -Q_in/V*X+mu_max*S/(S+K_S)*X

    return [dS, dX]


model = BaseModel('test', {})

model.fun_ode = ode_model_function

model.parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                    'S_in': 0.02, 'V': 20}

model.independent = ['t']
model._independent_values = {'t': np.linspace(0, 100, 5000)}
model._ordered_var = {'ode': ['S', 'X']}
model.initial_conditions = {'S': 0.02, 'X': 5e-5}

initial_val = [model.initial_conditions.get(var) for var in model._ordered_var['ode']]

solver = OdeSolver(model.fun_ode, model.initial_conditions.values(),
                   model._independent_values, (model.parameters,))

result = solver.solve(procedure="ode")


def test_model():

    assert_almost_equal(result[-1, 0], 0.0049996401543859316, decimal=10)
    assert_almost_equal(result[-1, 1], 0.010050542322437175, decimal=10)

if __name__ == "__main__":
    test_model()
