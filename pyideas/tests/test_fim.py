# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:45:57 2014

@author: timothy
"""

#general python imports
from __future__ import division
import pandas as pd
import numpy as np
from collections import OrderedDict

#bio-intense custom developments
from biointense import (AlgebraicModel, ParameterOptimisation, Measurements,
                        CalibratedConfidence)

from numpy.testing import assert_allclose


class TestFIM(object):

    def test_FIM(self):
        #Data
        data = np.array([[ 0., 2.3],
                         [20., 4.5],
                         [29., 6.6],
                         [41., 7.6],
                         [50., 9. ],
                         [65., 9.1],
                         [72., 9.4]])
        data = pd.DataFrame(data, columns = ['t','W']).set_index('t')
        M1measurements = Measurements(data)
        M1measurements.add_measured_errors({'W': 1.}, method='absolute')

        #Logistic

        parameters = {'W0':2.0805,
                      'Wf':9.7523,
                      'mu':0.0659}

        system = {'W':'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

        M1 = AlgebraicModel('model', system, parameters, ['t'])

        M1.independent = {'t': np.linspace(0, 72, 1000)}

        M1.run()

        M1optim = ParameterOptimisation(M1, M1measurements)
        M1optim.local_optimize()

        M1FIM = CalibratedConfidence(M1optim)
        FIM = M1FIM.get_FIM()

        FIM_expected = np.array([[6.47038949e+00, 2.21116827e+00, 2.64866592e+02],
                                 [2.21116827e+00, 2.74168793e+00, 1.54319463e+02],
                                 [2.64866592e+02, 1.54319463e+02, 1.45346865e+04]])
        assert_allclose(FIM, FIM_expected)
