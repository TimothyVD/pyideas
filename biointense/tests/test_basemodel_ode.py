# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:03:19 2015

@author: joris
"""
from __future__ import division

import numpy as np

from biointense.modelbase import BaseModel


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


model = BaseModel('test')

model.systemfunctions['ode'] = ode_model_function

model.parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                    'S_in': 0.02, 'V': 20}

model.independent_values = np.linspace(0, 100, 5000)
model.variables = {'ode': ['S', 'X']}
model.initial_conditions = {'S': 0.02, 'X': 5e-5}

from biointense.solver import OdeSolver

solver = OdeSolver(model)

result = solver.solve()

result['S'].plot()
result['X'].plot()


def test_model():

    assert result['S'].values[-1] == 0.0050000242650070203
    assert result['X'].values[-1] == 0.010049986013158058

if __name__ == "__main__":
    test_model()
