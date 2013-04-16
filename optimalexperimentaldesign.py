# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:39:49 2013

@author: VHOEYS
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
    def __init__(self,System, Parameters, Modelname = 'MyModel'):
        odegenerator.__init__(self, System, Parameters, Modelname = Modelname)
        
        self.criteria_optimality_info = {'A':'min', 'modA': 'max', 'D': 'max', 
                                         'E':'max', 'modE':'min'}
    
    
    def set_measured_times(self, meas_time):
        '''
        Based on the modelled timesteps, a subset is selected of measurable
        timesteps;
        
        Parameters
        -----------
        meas_time : 'all'|array
            if all is selected, all the modelled timesteps are included in the
            FIM calcluation        

        '''
        if meas_time == 'all':
            self._measdata_ID = self._Time
        else:
            print 'not yet implemented'
    
    def set_measured_errors(self, meas_error_dict, method = 'relative'):
        '''CURRENTLY UNITY MATRIX!!!!
        Measurement errors on the measured data; options are relative and
        absolute.

        The second component of the FIM is the measurement error in the form 
        of the inverse measurement error covariance matrix Q−1
        
        It should also be mentioned that correlations between measurements 
        can also be specified using this covariance matrix. Throughout this 
        package it will be assumed that the error characteristics of the 
        measurements can be described in a relatively simple way: by using 
        absolute or relative errors.
        
        Typically, Q is chosen as the inverse of the measurement error 
        covariance matrix(Marsili–Libelli et al., 2003; Omlin and Reichert, 
                             1999; Vanrolleghem and Dochain, 1998)
        
        Parameters
        -----------
        meas_error_dict : dict
            dictionary with the variable names and their corresponding errors
            (relative or absolute)
            
        method : relative|absolute
            relative is percentage value of the variable itself; absolute 
            is a constant measurement value
        
        References
        -----------
        De Pauw
        '''
        #compare with previous set measured variable; if same; ok, if not warning
        
        Meas_same = True
        for var in meas_error_dict:
            if var in self._Variables:
                if var in self._MeasuredList:
                    Meas_same = False
            else:
                raise Exception('%s is not a variable in the current model' %var)
        if Meas_same == False or len(meas_error_dict) <> len(self._Variables):
            print 'Measured variables are updated!'
            self.set_measured_states(meas_error_dict.keys())
        self.Meas_Errors = collections.OrderedDict(sorted(meas_error_dict.items(), key=lambda t: t[0]))
        
        #update measurement error covariance matrix eenheidsmatrix
        self.Qerr = np.identity(len(self._MeasuredList))
        
        #recalculate values of error percentages towards values
#        self.Qerr = np.diag(self.Meas_Errors.values()) #np.diag_indices_from()
    
    def get_FIM(self):
        '''
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

            #GET ERROR_MATRIX!! TODO
                            
            #calculate matrices
            FIMt = np.matrix(sensmatrix).transpose() * np.matrix(self.Qerr) * np.matrix(sensmatrix)
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
              
    def set_measurement_data(self):
        '''
        Give measurement data to calculate objective functions and perform
        optimizations
        '''
        pass    
    
    def residual_analysis(self):
        pass
    
    
    
    
    