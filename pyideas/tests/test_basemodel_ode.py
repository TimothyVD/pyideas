# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:03:19 2015

@author: joris
"""
from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal

from biointense import Model

system = {'dS': 'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
          'dX': '-Q_in/V*X+mu_max*S/(S+K_S)*X'}

parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
              'S_in': 0.02, 'V': 20}

M1 = Model('ode', system, parameters)

M1.independent = {'t': np.linspace(0, 100, 5000)}
M1.initial_conditions = {'S': 0.02, 'X': 5e-5}


result = M1._run()


def test_model():

    assert_almost_equal(result[-1, 0], 0.0049996401543859316, decimal=5)
    assert_almost_equal(result[-1, 1], 0.010050542322437175, decimal=5)

if __name__ == "__main__":
    test_model()
