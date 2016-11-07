# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:30:34 2015

@author: joris
"""
from __future__ import division

import numpy as np

from pyideas.modelbase import BaseModel


def alg_model_function(independent, parameters):
    t = independent['t']

    W0 = parameters['W0']
    Wf = parameters['Wf']
    mu = parameters['mu']

    W = W0*Wf/(W0 + (-W0 + Wf)*np.exp(-mu*t)) + np.zeros(len(t))

    algebraic = np.array([W]).T

    return algebraic

parameters = {'W0': 2.0805,
              'Wf': 9.7523,
              'mu': 0.0659}

model = BaseModel('test', parameters, ['W'],  ['t'], alg_model_function)

model.independent = {'t': np.linspace(0, 72, 1000)}

result = model._run()


def test_model():
    assert result[-1, 0] == 9.4492688322077534


if __name__ == "__main__":
    test_model()
