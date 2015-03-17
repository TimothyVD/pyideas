# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:30:34 2015

@author: joris
"""
from __future__ import division

import numpy as np

from biointense.modelbase import BaseModel


def alg_model_function(independent, parameters):
    t = independent['t']

    W0 = parameters['W0']
    Wf = parameters['Wf']
    mu = parameters['mu']

    W = W0*Wf/(W0 + (-W0 + Wf)*np.exp(-mu*t)) + np.zeros(len(t))

    algebraic = np.array([W]).T

    return algebraic

model = BaseModel('test', {})
#model.systemfunctions['algebraic'] = alg_model_function
model.fun_alg = alg_model_function
model.parameters = {'W0': 2.0805,
                    'Wf': 9.7523,
                    'mu': 0.0659}

model.independent = ['t']
model._independent_values = {'t': np.linspace(0, 72, 1000)}
model._ordered_var = {'algebraic': ['W']}


from biointense.solver import AlgebraicSolver

solver = AlgebraicSolver(model)
result = solver.solve()
result.plot()


def test_model():
    assert result['W'].values[-1] == 9.4492688322077534


if __name__ == "__main__":
    test_model()