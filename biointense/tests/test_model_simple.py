# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 13:30:43 2015

@author: joris
"""
from __future__ import division

import numpy as np
import pandas as pd

from pandas.util.testing import assert_frame_equal

from biointense import Model


# 1) simple algebraic state space variable of interest

def test_alg_state():

    system = {'dP': 'k1 * P',
              'dS': 'k2 * S',
              'A': 'P + S'}
    parameters = {'k1': -0.2, 'k2': 1.0}

    model = Model('simple_test', system, parameters)

    x = np.linspace(0, 10, 11)
    model.initial_conditions = {'P': 50, 'S': 10}
    model.independent = {'t': x}

    result = model.run()

    # expected output (analytical solution)
    S = 10 * np.exp(1.0*x)
    P = 50 * np.exp(-0.2*x)

    expected = pd.DataFrame({'P': P, 'S': S}, index=x)
    expected['A'] = expected['S'] + expected['P']
    expected.index.names = ['t']

    assert_frame_equal(result, expected.reindex(columns=result.columns))


# 2) simple algebraic substitution


def test_alg_substitution():

    system = {'dP': 'k1 * P + A',
              'dS': 'k2 * S',
              'A': 'S + P'}
    parameters = {'k1': -0.5, 'k2': 1.0}

    model = Model('simple_test', system, parameters)
    x = np.linspace(0, 10, 11)
    model.initial_conditions = {'P': 50, 'S': 10}
    model.independent = {'t': x}

    result = model.run()

    # expected output (analytical solution)
    k1, k2 = -0.5, 1.0
    S = 10 * np.exp(k2*x)
    P = 10 * (np.exp((k1 + 1)*x) - np.exp((k2)*x)) / (k1 - k2 + 1) \
        + 50 * np.exp((k1 + 1)*x)

    expected = pd.DataFrame({'S': S, 'P': P}, index=x)
    expected['A'] = expected['S'] + expected['P']
    expected.index.names = ['t']

    assert_frame_equal(result, expected.reindex(columns=result.columns))
