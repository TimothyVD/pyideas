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

from plotfunctions import *



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
    def __init__(self):
        odegenerator.__init__(self, System, Parameters, Modelname = 'MyModel')
    
    
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
        '''CURRENTLY UNITY
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
        self.Meas_Errors = collections.OrderedDict(sorted(meas_error_dict .items(), key=lambda t: t[0]))
        
        #update measurement error covariance matrix eenheidsmatrix
        self.Qerr = np.identity(len(self._MeasuredList))
        
        #recalculate values of error percentages towards values
#        self.Qerr = np.diag(self.Meas_Errors.values()) #np.diag_indices_from()
    
    def FIM(self):
        '''
        Based on the measurement errors and timesteps to include evaluation,
        the FIM is calculated
        
        Notes
        ------
        
        '''
        FIM = np.empty((len(self.Parameters),len(self.Parameters)))
        
        for timestep in self._measdata_ID:
            #GET SENS_MATRIX
            sensmatrix = np.empty((len(self._MeasuredList),len(self.Parameters)))
            #create sensitivity amtrix
            varcounter = 0
            for var in self.numerical_sensitivity:
                sensmatrix[varcounter,:] = np.array(self.numerical_sensitivity[var].xs(timestep))
                varcounter+=1

            #GET ERROR_MATRIX!! TODO
                            
            #calculate matrices
            FIMt = np.matrix(sensmatrix).transpose() * np.matrix(self.Qerr) * np.matrix(sensmatrix)
            FIM+=FIMt
            
    def D_criterium(self):
        pass
    


    def set_measurement_data(self):
        '''
        Give measurement data to calculate objective functions and perform
        optimizations
        '''
        pass    
    
    def residual_analysis(self):
        pass
    
    
    
    
    