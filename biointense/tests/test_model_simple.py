# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:30:43 2015

@author: joris
"""
from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal
from pandas.util.testing import assert_frame_equal

from biointense.model import Model
from biointense.solver import OdeSolver


## 1) simple algebraic state space variable of interest

def test_alg_state():


    system = {'dP': 'k1 * P',
              'dS': 'k2 * S',
              'A': 'P + S'}
    parameters = {'k1': -0.2, 'k2': 1.0}

    model = Model('simple_test', system, parameters)

    x = np.linspace(0, 9, 10)
    model.independent_values = x
    model.set_initial({'P': 50, 'S': 10})

    solver = OdeSolver(model)
    result = solver.solve()

    # expected output
    p = 50
    P = []

    for i in range(10):
        p = p - 0.2*p
        P.append(p)

    expected = pd.DataFrame({'P': P, 'S': np.linspace(20, 110, 10)},
                             index=x)
    expected['A'] = expected['S'] + expected['P']

    assert_frame_equal(result, expected)



## 2) simple algebraic substitution

def test_alg_substitution():

    system = {'dP': 'k1 * P + A',
              'dS' : 'k2 * S',
              'A': 'S + P'}
    parameters = {'k1': -0.5, 'k2': 1.0}

    model = Model('simple_test', system, parameters)
    model.independent_values = np.linspace(0, 9, 10)
    model.set_initial({'P': 50, 'S': 10})

    solver = OdeSolver(model)
    result = solver.solve()

    # expected output
    S = np.linspace(10, 110, 11)
    p = 50
    P = []

    for i in range(10):
        p = p - 0.5*p + (S[i] + p)
        P.append(p)

    expected = pd.DataFrame({'S': np.linspace(20, 110, 10),
                             'P': P})
    expected['A'] = expected['S'] + expected['P']

    assert_frame_equal(result, expected)


## 3) algebraic equation as constraint -> not yet implemented
