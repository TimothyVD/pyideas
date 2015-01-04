# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:30:34 2015

@author: joris
"""
from __future__ import division

import unittest

import numpy as np

from biointense.model import Model


class TestAlgebraicModel(unittest.TestCase):

    def SetUp(self):

        system = {'algebraic': {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}}
        parameters = {'W0': 2.0805,
                      'Wf': 9.7523,
                      'mu': 0.0659}

        model = Model('Modsim1', system, parameters)

        model.independent_values = np.linspace(0, 72, 1000)
        #model.variables = {'algebraic': ['W']}

        self.model

    def test_model_run(self):

        result = self.model.run()
        assert result['W'].values[-1] == 9.4492688322077534
