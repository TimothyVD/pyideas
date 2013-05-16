# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:39:49 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator package
"""

from __future__ import division
import sys
import os
import numpy as np
import sympy

import pandas as pd

if sys.hexversion > 0x02070000:
    import collections
else:
    import ordereddict as collections

from plotfunctions import *
from ode_generator import odegenerator
 
        
class OED(odegenerator):
    '''
    OED aimed class functioning
    
    Deliverables:
    - FIM calcluation based on odegenerator model sensitivities
    - Error covariance options (relative and absolute)
    - Measurement timestep selection to support FIM calclulations
    - link with optimization algorithms for optimization of parameters/designs
    
    AIM:
    - Link with robust OED -> sequential design (paropt - exp-opt - paropt)
    '''
    
    def __init__(self,System, Parameters, Modelname = 'MyModel'): #Measurements
        '''
        Measurements: measurements-class instance
        '''
        odegenerator.__init__(self, System, Parameters, Modelname = Modelname)
        
        self.criteria_optimality_info = {'A':'min', 'modA': 'max', 'D': 'max', 
                                         'E':'max', 'modE':'min'}
        
        #measurements stuff
        #___________________
        #control for corresponding names with the model
        #set measured timesteps, measured variables and self.Q

    def get_mod_times(self):
        '''view timesteps of model output
        '''
        return self._Time
    
    def control_measmod_time(self):
        '''
        '''
        pass


    def get_FIM(self):
        '''
        DEPRECIATED!!
        Needs update to be able to manage different measurment length types
        in order ot be combined with the current Error_Covariance_Method        
        
        Based on the measurement errors and timesteps to include evaluation,
        the FIM is calculated
        
        Notes
        ------
        
        '''
        
        try:
            self._measdata_ID
        except:
            self.set_measured_times('all')
            print 'all model timesteps are included in FIM calculation'
        
        #test for sensitivity
        try:
            self.numerical_sensitivity
        except:
            self.numeric_local_sensitivity()
        
        
        self.FIM = np.zeros((len(self.Parameters),len(self.Parameters)))
        for timestep in self._measdata_ID:
            #GET SENS_MATRIX
            sensmatrix = np.zeros((len(self._MeasuredList),len(self.Parameters)))
            #create sensitivity amtrix
            varcounter = 0
            for var in self._MeasuredList:
                sensmatrix[varcounter,:] = np.array(self.numerical_sensitivity[var].xs(timestep))
                varcounter+=1

            #GET ERROR_MATRIX!! TODO!!
    
            #calculate matrices
            FIMt = np.matrix(sensmatrix).transpose() * np.linalg.inv(np.matrix(self.Qerr)) * np.matrix(sensmatrix)
            self.FIM = self.FIM + FIMt
            
        return self.FIM

    def _check_for_FIM(self):
        ''' help function for FIM testing
        If FIM exists, ok, otherwise start FIM calculation
        '''  
        try:
            self.FIM
        except:
            print 'FIM matrix is calculated...'
            self.get_FIM()
            print '... done!'

    def A_criterium(self):
        '''OED design A criterium
        With this criterion, the trace of the inverse of the FIM is minimized, 
        which is equivalent to minimizing the sum of the variances of the 
        parameter estimates. In other words, this criterion minimizes the 
        arithmetic average of the variances of the parameter estimate. 
        Because this criterion is based on an inversion of the FIM, 
        numerical problems will arise when the FIM is close to singular.        
        '''
        self._check_for_FIM()
        print 'MINIMIZE A criterium for OED'
        return self.FIM.I.trace()
        
    def modA_criterium(self):
        '''OED design modified A criterium
        With this criterion, the trace of the inverse of the FIM is minimized, 
        which is equivalent to minimizing the sum of the variances of the 
        parameter estimates. In other words, this criterion minimizes the 
        arithmetic average of the variances of the parameter estimate. 
        Because this criterion is based on an inversion of the FIM, 
        numerical problems will arise when the FIM is close to singular.        
        '''
        self._check_for_FIM()
        print 'MAXIMIZE modA criterium for OED'
        return self.FIM.trace()                   
        
    def D_criterium(self):
        '''OED design D criterium
        Here, the idea is to maximize the determinant of the FIM 
        (Box and Lucas, 1959). The latter is inversely proportional to the 
        volume of the confidence region of the parameter es- timates, and this 
        volume is thus minimized when maximizing det (FIM). In other words, 
        one minimizes the geometric average of the variances of the parameter 
        estimates. More- over, D-optimal experiments possess the property of 
        being invariant with respect to any rescaling of the parameters 
        (Petersen, 2000; Seber and Wild, 1989). According to Walter and 
        Pronzato (1997), the D-optimal design criterion is the most used 
        criterion. However, several authors have pointed out that this 
        criterion tends to give excessive importance to the parameter which 
        is most influential.
        '''
        self._check_for_FIM()
        print 'MAXIMIZE D criterium for OED'
        return np.linalg.det(self.FIM)          

    def E_criterium(self):
        '''OED design E criterium
        The E-optimal design criterion maximizes the smallest eigenvalue of 
        the FIM and thereby minimizes the length of the largest axis of the 
        confidence ellipsoid. Thus, these designs aim at minimizing the 
        largest parameter estimation variance and thereby at maximizing the 
        distance from the singular, unidentifiable case.
        '''
        self._check_for_FIM()
        print 'MAXIMIZE E criterium for OED'
        w, v = np.linalg.eig(self.FIM)
        return min(w)

    def modE_criterium(self):
        '''OED design modE criterium
        With this criterion, the focus is on the minimization of the condition 
        number, which is the ratio between the largest and the smallest 
        eigenvalue, or, in other words, the ratio of the shortest and the 
        longest ellipsoid axes. The minimum of this ratio is one, which 
        corresponds to the case where the shape of the confidence ellipsoid 
        is a (hyper)sphere.
        '''
        self._check_for_FIM()
        print 'MINIMIZE modE criterium for OED'
        w, v = np.linalg.eig(self.FIM)
        return max(w)/min(w)


    def get_all_optimality_design_criteria(self):
        '''Return all optimality criteria
        
        '''
        self._check_for_FIM()
        self._all_crit = {'A':self.A_criterium(), 'modA': self.modA_criterium(), 
                          'D': self.D_criterium(), 'E':self.E_criterium(), 'modE':self.modE_criterium()}
        return self._all_crit

    
    
    
    