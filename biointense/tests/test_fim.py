# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:45:57 2014

@author: timothy
"""

#general python imports
from __future__ import division
import pandas as pd
from collections import OrderedDict

#bio-intense custom developments
from biointense import *

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
        data = pd.DataFrame(data, columns = ['time','W']).set_index('time')
        measurements = ode_measurements(data)

        #Logistic

        Parameters = {'W0':2.0805,
                      'Wf':9.7523,
                      'mu':0.0659}

        Alg = {'W':'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

        M1 = DAErunner(Parameters = Parameters, Modelname ='Modsim1', Algebraic = Alg)

        M1.set_xdata({'start':0,'end':72,'nsteps':1000})
        M1.set_measured_states(['W'])

        M1.solve_algebraic(plotit = False)

        optim1 = ode_optimizer(M1, measurements)
        optim1.local_parameter_optimize()

        FIM_stuff1 = ode_FIM(optim1)
        FIM = FIM_stuff1.get_newFIM()

        FIM_expected = np.array([[6.47038949e+00, 2.21116827e+00, 2.64866592e+02],
                                 [2.21116827e+00, 2.74168793e+00, 1.54319463e+02],
                                 [2.64866592e+02, 1.54319463e+02, 1.45346865e+04]])
        assert_allclose(FIM, FIM_expected)

    def test_slow_fast_FIM(self):
        #Data
        data = np.array([[ 0., 2.3],
                         [20., 4.5],
                         [29., 6.6],
                         [41., 7.6],
                         [50., 9. ],
                         [65., 9.1],
                         [72., 9.4]])
        data = pd.DataFrame(data, columns = ['time','W']).set_index('time')
        measurements = ode_measurements(data)

        #Logistic

        Parameters = {'W0':2.0805,
                      'Wf':9.7523,
                      'mu':0.0659}

        Alg = {'W':'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

        M1 = DAErunner(Parameters = Parameters, Modelname ='Modsim1', Algebraic = Alg)

        M1.set_xdata({'start':0,'end':72,'nsteps':1000})
        M1.set_measured_states(['W'])

        M1.solve_algebraic(plotit = False)

        optim1 = ode_optimizer(M1, measurements)

        FIM_stuff1 = ode_FIM(optim1)

        assert_allclose(FIM_stuff1.get_newFIM(), FIM_stuff1.get_FIM())
