# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:33:14 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator linkfile for optimization
"""

from __future__ import division
import warnings
import sys
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if sys.hexversion > 0x02070000:
    import collections
else:
    import ordereddict as collections

from plotfunctions import *
from ode_generator import odegenerator
from optimalexperimentaldesign import OED

import matplotlib.pyplot as plt


class MeasData(object):
    '''class to include measured data in the framework
    
    
    Input possibilities (cfr. FME package R):
        
        1. Dictionary: {'variables':[var of measurement], 'time': [timeindex of measurement], 'values':[measurements], 'errors':[errors of meas]}
        2. DataFrame of the above Dictionary
        3. Dictionary: {'time': [time indices], 'var1': [meas for the timesteps of var1], 'var2':[meas for the timesteps of var2]}
        4. DataFrame of the above  
    
    ATTENTION: The 'variables' and 'values' is required to let it work!
        
    '''
    
    def __init__(self, measdata):
        '''
        '''
        if isinstance(measdata, dict):            
            #Different variables names in key-names
            if 'variables' in measdata:
                if not 'values' in measdata:
                    raise Exception('Values and variables needed as key-value')
                if not 'time' in measdata:
                    raise Exception('Time information of the measurements lacking!')
                self.Data = pd.DataFrame(measdata).pivot(index='time', columns='variables', values='values')
                
            else: #all variables in key names, so same length in all arrays
                lencheck = np.array([len(value) for value in measdata.items()])
                if not lencheck.min() == lencheck.max():
                    raise Exception('Dictionary data input requires equal lengths in this format')          
                
                if 'time' in measdata:
                    #convert Dictionary to pandas DataFrame 
                    indext=measdata['time']
                    measdata = {key: value for key, value in measdata.items() if key != 'time'}
                    self.Data = pd.DataFrame(measdata, index=indext)  
                else:
                    self.Data = pd.DataFrame(measdata)
                    print 'Attention: Time is not explicitly set!'

        elif isinstance(measdata, pd.DataFrame):            
            #check if time is a column name
            if 'time' in t1.columns:
                if 'variables' in measdata:
                    self.Data = measdata.pivot(index='time', columns='variables', values='values')
                else:
                    self.Data = measdata.set_index('time')                                
            else:
                print 'index of dataframe is seen as measurement time steps, and colnames are the measured variables'
    
        else:
            raise Exception('Measured Data must be added as pandas DataFrame or dictionary of np arrays')
        
        #We provide for internal purposes also second data-type: dict with {'cvar1':Timeserie,'var': Timeserie}                
        self._data2dictsystem()
        self.get_measured_variables()                  

    def _data2dictsystem(self):
        '''
        Help function to merge the different date-profiles
        '''
        self.Data_dict={}
        for var in self.Data.columns:
            self.Data_dict[var] = pd.DataFrame(self.Data[var].dropna())
            
    def add_measured_variable(self, newdata):
        '''add an extra measurement
        
        Parameters
        ------------
        newdata :  pd.DataFrame or dict
            Variable and the corresponding timesteps meadured
        
        Notes
        -----
        Example dict: {'time':[0.5,2.,8.],'DO2':[5.,9.,2.]}
        
        '''
        if isinstance(newdata, dict):
            newdata = pd.DataFrame(newdata).set_index('time')
            self.Data = pd.concat([self.Data, newdata], axis=1).sort()                       
        elif isinstance(newdata, pd.DataFrame):
            self.Data = pd.concat([self.Data, newdata], axis=1).sort()           

        #update the datadictionary
        self._data2dictsystem()  
        self.get_measured_variables()                  

    def get_measured_times(self):
        '''
        Based on the measurements timesteps, an array is made with the 
        timesteps to get modelled output from (over all variables)    

        '''
        self._measured_timesteps = np.array(self.Data.index).astype('float')
        return self._measured_timesteps

    def get_measured_variables(self):
        '''
        Based on the measurements, get the variable names measured
        '''
        self._measured_variables = self.Data.columns.tolist()
        return self._measured_variables
        
        
    def add_measured_errors(self, meas_error_dict, method = 'relative',
                            lower_accuracy_bound = None, 
                            minimal_relative_error = None):
        r'''calculates standard deviation of measurements
        
        Measurement errors on the measured data; options are relative and
        absolute.

        The second component of the FIM is the measurement error in the form 
        of the inverse measurement error covariance matrix Q−1
        
        It should also be mentioned that correlations between measurements 
        can also be specified using this covariance matrix. Throughout this 
        package it will be assumed that the error characteristics of the 
        measurements can be described in a relatively simple way:
            by asolute or relative errors or by Ternbach 
        description.
        
        Typically, Q is chosen as the inverse of the measurement error 
        covariance matrix(Marsili–Libelli et al., 2003; Omlin and Reichert, 
        1999; Vanrolleghem and Dochain, 1998)
        
        Error values are also added to the data_dict version. 
        
        Parameters
        -----------
        meas_error_dict : dict
            dictionary with the variable names and their corresponding errors
            (relative or absolute)
        method : relative|absolute
            relative is percentage value of the variable itself; absolute 
            is a constant measurement value

        Notes
        -----
        For the Ternbach method,  the standard deviations of the measurements 
        were calculated by:
        
        .. math:: \sigma_y = \hat{y} \cdot \varsigma_y \cdot \left(1+\frac{1}{(\frac{\hat{y}}{lb_y})^2 + \frac{\hat{y}}{lb_y}} \right)
        
        Here, :math:`\varsigma_y` and :math:`lb_y` respectively represent a 
        constant minimal relative error and a lower accuracy bound on the 
        measurement of y. In this way, the standard deviations of the 
        meaurements are proportional to the value of the measurements
        :math:`\hat{y}`.        
        '''
        try:
            self._Error_Covariance_Matrix
            warnings.warn('Measurement errors are updated ', UserWarning)
        except:
            print 'Error Covariance Matrix is created'
        
        for var in meas_error_dict:
            if not var in self.get_measured_variables():       
                raise Exception('Variable ', var, ' not listed in current measurements.')

        self.Meas_Errors = collections.OrderedDict(sorted(meas_error_dict.items(), key=lambda t: t[0]))                    
        self.Meas_Errors_type = method
        
        #update measurement error covariance matrix eenheidsmatrix
        #self._Error_Covariance_Matrix = np.identity(len(self._MeasuredList))
        #OLD VERSION; only for equal measurement size:
        #self._Error_Covariance_Matrix = np.zeros((len(self.get_measured_variables()), 
        #len(self.get_measured_variables()), len(self.get_measured_times())))
       
        #The number of available measurements is time-dependent, so the FIM
        #and WSSE calculation need adaptation. In order to preserve flexibility
        #We'll put each timestep in a separate item in a dictionary
        self._Error_Covariance_Matrix = {}
        
        if method == 'absolute':
            for var in self.Meas_Errors:
                measerr = self.Meas_Errors[var]
                self.Data_dict[var]['error'] = measerr**2.
            
            for timestep in self.get_measured_times():   
                self._Error_Covariance_Matrix[timestep] = np.zeros((len(self.Data.ix[timestep].dropna()), len(self.Data.ix[timestep].dropna())))
                for ide, var in enumerate(self.Data.ix[timestep].dropna().index):
                    measerr = self.Meas_Errors[var]
                    self._Error_Covariance_Matrix[timestep].values[ide,ide] = measerr**2.  #De 1/sigma^2 komt bij inv berekening van FIM
                
        elif method == 'relative':   
            for var in self.Meas_Errors:
                measerr = self.Meas_Errors[var]
                self.Data_dict[var]['error'] = np.array((measerr*self.Data_dict[var][var])**2.).flatten()
            
            for timestep in self.get_measured_times():  
                print timestep,' timestep'
                temp = np.zeros((len(self.Data.ix[timestep].dropna()), len(self.Data.ix[timestep].dropna())))
                self._Error_Covariance_Matrix[timestep] = pd.DataFrame(temp, index=self.Data.ix[timestep].dropna().index.tolist(), columns=self.Data.ix[timestep].dropna().index.tolist())
                for ide, var in enumerate(self.Data.ix[timestep].dropna().index):
                    print 'var',var
                    print ide,'ide'
                    measerr = self.Meas_Errors[var]
                    self._Error_Covariance_Matrix[timestep].values[ide,ide] = np.array((measerr*self.Data_dict[var].ix[timestep][var])**2.)#.flatten()
                
        elif method == 'Ternbach': #NEEDS CHECKUP!!
            for var in self.Meas_Errors:
                yti = self.Data_dict[var][var]
                measerr = self.Meas_Errors[var]      
                temp=1.+ 1./((yti/lower_accuracy_bound)**2 +(yti/lower_accuracy_bound))
                self.Data_dict[var]['error'] = yti*minimal_relative_error*temp
            
            for timestep in self.get_measured_times():   
                self._Error_Covariance_Matrix[timestep] = np.zeros((len(self.Data.ix[timestep].dropna()), len(self.Data.ix[timestep].dropna())))
                for ide, var in enumerate(self.Data.ix[timestep].dropna().index):
                    yti = self.Data_dict[var].ix[timestep][var]
                    measerr = self.Meas_Errors[var]      
                    temp=1.+ 1./((yti/lower_accuracy_bound)**2 +(yti/lower_accuracy_bound))
                    self._Error_Covariance_Matrix[timestep].values[ide,ide] = yti*minimal_relative_error*temp 

import datetime
class ModOptim_saver():
    '''
    
    Better would be to make this as baseclass and add the other part on this one
    '''
    def __init__(self):
        self.info='Saving of output settings and fit characteristics on ',datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    

class ModOptim(object):
    '''class to include measured data in the framework
    
    Last run is always internally saved...
    
    '''
    
    def __init__(self, odeModel, Data):
        '''
        '''
        #check inputs
        if not isinstance(odeModel, odegenerator) and isinstance(odeModel, OED):
            raise Exception('Bad input type for model or oed')
        if not isinstance(Data, MeasData):
            raise Exception('Bad input type for Data')
            

        #compare with previous set measured variable; if same; ok, if not warning
        Meas_same = True
        for var in Data.get_measured_variables():
            if var in odeModel._Variables:
                if not var in odeModel._MeasuredList:
                    Meas_same = False
            else:
                raise Exception('%s is not a variable in the current model' %var)
        if Meas_same == False or len(Data.get_measured_variables()) <> len(odeModel._Variables):
            print 'Measured variables are updated!'
            Dat.set_measured_states(Data.get_measured_variables())
        
        self.Data = Data.Data
#        self.Data.columns = [var+'_meas' for var in self.Data.columns]
        self._data = Data
        self._data_dict = Data.Data_dict
        self._model = odeModel
        #create initial set of information:
        self._solve_for_opt()
        self.get_WSSE()

    def _parmapper(self, pararray):
        '''converts parameter array for minimize function in dict
        
        Gets an array of values and converts into dictionary
        '''
        # A FIX FOR CERTAIN SCIPY MINIMIZE ALGORITHMS
        try:
            pararray = pararray[0,:]   
        except:
            pararray = pararray
            
        pardict = self._get_fitting_parameters()
        for i, key in enumerate(pardict):
            pardict[key] = pararray[i]
        return pardict

    def _pardemapper(self, pardict):
        '''converts parameter dict in array for minimize function
        
        Gets an array of values and converts into dictionary
        '''
        #compare with existing pardict
        if sorted(pardict.keys()) != sorted(self._get_fitting_parameters().keys()):
            raise Exception('The fitting parameters are not equal with the given dict')
        
        pararray = np.zeros(len(self._get_fitting_parameters()))
        for i, key in enumerate(pardict):
            pararray[i] = pardict[key]
        return pararray
        

    def _solve_for_opt(self, parset=None):
        '''
        ATTENTION: Zero-point also added, need to be excluded for optimization
        '''
        #run option        
        if parset != None:
            #run model first with new parameters
            for par in self._get_fitting_parameters().keys():
                self._model.Parameters[par] = parset[par]
        
        self._model._Time = np.concatenate((np.array([0.]),self._data.get_measured_times()))
        self.ModelOutput = self._model.solve_ode(plotit=False)
#        self.ModelOutput.columns = [var+'_model' for var in self.ModelOutput.columns]
        self._model.set_time(self._model._TimeDict)
        #put ModMeas in set
        self.ModMeas = pd.concat((self.Data,self.ModelOutput), axis=1, keys=['Measured','Modelled'])        
        return self.ModelOutput
              
    def _solve_for_visual(self, parset=None):
        '''
        ATTENTION: Zero-point also added, need to be excluded for optimization
        '''
        #run option        
        if parset != None:
            #run model first with new parameters
            for par in self._get_fitting_parameters().keys():
                self._model.Parameters[par] = parset[par]
                
        visual_ModelOutput = self._model.solve_ode(plotit=False)
        return visual_ModelOutput     
        
              
    def get_WSSE(self, pararray=None):
        '''calculate weighted SSE

        '''
        if pararray != None:
            self._solve_for_opt(parset = self._parmapper(pararray))
        else:
            self._solve_for_opt()
        
        #Residuals for the current model_output
        self.residuals = (self.ModelOutput-self.Data).dropna(how='all') 
        self.unweigthed_SSE = (self.residuals**2).sum() 
        
        #WSSE CALCULATION       
        #sum over the timesteps (order is not important, so dict-iteration could also be used)
        self.WSSE = 0.0
        for timestep in self._data.get_measured_times():
            resid = np.matrix(self.residuals.ix[timestep].dropna().values)
            qerr = np.matrix(self._data._Error_Covariance_Matrix[timestep])
            self.WSSE = self.WSSE + resid * np.linalg.inv(qerr)*resid.transpose()
        self.WSSE = np.array(self.WSSE)            
        return self.WSSE

            
    def plot_ModMeas(self):
        '''plot outputs
        '''        
        fig,axes = plt.subplots(len(self.Data.columns),1)
        for i,var in enumerate(self.Data.columns):
            axes[i].plot(self.Data.index, self.Data[var], marker='o', linestyle='none', color='k')
            axes[i].plot(self._solve_for_visual().index, self._solve_for_visual()[var], linestyle='--', color='k')
        
    def get_all_parameters(self):
        '''
        '''
        return self._model.Parameters

    def set_fitting_parameters(self):
        '''
        '''
        pass #currently all fitting parameters

    def _get_fitting_parameters(self):
        '''
        '''
        return self._model.Parameters

    
    def optimize(self, initial_parset=None, add_plot=True):
        '''find parameters for optimal fit
        
        initial_parset: dict!!
        
        '''
        #first save the output with the 'old' parameters
        #if initial parameter set given, use this, run and save   
        self._Pre_Optimize = ModOptim_saver()
        if initial_parset != None:
            parray = self._pardemapper(initial_parset)
            self._solve_for_opt(initial_parset)
            self._Pre_Optimize.parameters = initial_parset
            self._Pre_Optimize.visual_output = self._solve_for_visual(initial_parset)
        else:
            parray = self._pardemapper(self._get_fitting_parameters())
            self._solve_for_opt()
            self._Pre_Optimize.parameters = self._get_fitting_parameters()
            self._Pre_Optimize.visual_output = self._solve_for_visual(self._get_fitting_parameters())
        #Save them for comparison
        self._Pre_Optimize.residuals = self.residuals
        self._Pre_Optimize.ModMeas = self.ModMeas
        self._Pre_Optimize.WSSE = self.WSSE
        self._Pre_Optimize.unweigthed_SSE = self.unweigthed_SSE

        #OPTIMIZATION
        #TODO: ADD OPTION FOR SAVING THE PARSETS (IN GETWSSE!!)
        #different algorithms: but implementation  Anneal and CG are not working 
        #a first fix made Powell work
        res = minimize(self.get_WSSE, parray, method= 'Anneal')
        
        #comparison plot
        fig,axes = plt.subplots(len(self.Data.columns),1)
        for i,var in enumerate(self.Data.columns):
            #plot data
            axes[i].plot(self.Data.index, self.Data[var], marker='o', linestyle='none', color='k', label='Measured')
            #plot output old
            axes[i].plot(self._Pre_Optimize.visual_output.index, self._Pre_Optimize.visual_output[var], linestyle='-.', color='k', label='No optimization')            
            #plot output new
            axes[i].plot(self._solve_for_visual().index, self._solve_for_visual()[var], linestyle='--', color='k', label='Optimized')
        return res
        
        
        
        











        #Create Multi-Index -> not working
#        arrays = [['Model']*len(self._data.get_measured_variables())+['Meas']*len(self._data.get_measured_variables()),self._data.get_measured_variables()*2]
#        tuples = zip(*arrays)
#        index = pd.MultiIndex.from_tuples(tuples, names=['Type', 'Variables'])
                