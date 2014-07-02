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
from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises


class TestFIM(object):
    @classmethod
    def setup_class(klass):
        """This method is run once for each class before any tests are run"""

    @classmethod
    def teardown_class(klass):
        """This method is run once for each class _after_ all tests are run"""

    def setUp(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

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
        
        M1.set_time({'start':0,'end':72,'nsteps':1000})
        M1.set_measured_states(['W'])
        
        M1.solve_algebraic(plotit = False)
                
        optim1 = ode_optimizer(M1, measurements)
        optim1.local_parameter_optimize()
        
        FIM_stuff1 = ode_FIM(optim1)
        FIM = FIM_stuff1.get_newFIM()
        
        assert_equal(FIM.any(), 
                     np.array([[  6.47038949e+00,   2.21116827e+00,   2.64866592e+02],
                               [  2.21116827e+00,   2.74168793e+00,   1.54319463e+02],
                               [  2.64866592e+02,   1.54319463e+02,   1.45346865e+04]]).any())
                               
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
        
        M1.set_time({'start':0,'end':72,'nsteps':1000})
        M1.set_measured_states(['W'])
        
        M1.solve_algebraic(plotit = False)
                
        optim1 = ode_optimizer(M1, measurements)
        
        FIM_stuff1 = ode_FIM(optim1)
       
        assert_equal(FIM_stuff1.get_newFIM().any(), FIM_stuff1.get_FIM().any())