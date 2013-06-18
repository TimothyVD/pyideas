# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:39:49 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator package
"""

from __future__ import division
import sys
import os
import datetime
import numpy as np
import sympy

import pandas as pd
from scipy import optimize

if sys.hexversion > 0x02070000:
    import collections
else:
    import ordereddict as collections

from plotfunctions import *
from matplotlib.ticker import MaxNLocator
from ode_generator import odegenerator
from measurements import ode_measurements

class ode_optim_saver():
    '''
    Better would be to make this as baseclass and add the other part on this one,
    but this does the job
    '''
    def __init__(self):
        self.info='Last saving of output settings and fit characteristics on ',datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    

class ode_optimizer(object):
    '''class to include measured data in the framework
    
    Last run is always internally saved...
    
    '''
    
    def __init__(self, odeModel, Data):
        '''
        '''
        #check inputs
        if not isinstance(odeModel, odegenerator):
            raise Exception('Bad input type for model or oed')
        if not isinstance(Data, ode_measurements):
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
            print 'Measured variables are updated in model!'
            odeModel.set_measured_states(Data.get_measured_variables())
        
        self.Data = Data.Data
#        self.Data.columns = [var+'_meas' for var in self.Data.columns]
        self._data = Data
        self._data_dict = Data.Data_dict
        self._model = odeModel
        
        #create initial set of information:
        self._solve_for_opt()
        self.get_WSSE()
        #All parameters are set as fitting
        self.set_fitting_parameters(self._model.Parameters)

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

    def set_pars_GUI(self):
        '''GUI based parameter adaption
        '''
        try:
            from formlayout import fedit
        except:
            raise Exception("Module formlayout is needed to do interactive adjustment")
        
        parlist = []
        for key in self._model.Parameters:
            parlist.append((key,self._model.Parameters[key]))
        
        newparlist = fedit(parlist, title="Update your aprameter values",
                               comment="Give your new <b>parameter values</b>")
        
#        print "result:", newparlist
        newparlistdict = self._parmapper(newparlist)
        for par in newparlistdict:
            self._model.Parameters[par] = newparlistdict[par]        
        

    def _solve_for_opt(self, parset=None):
        '''
        ATTENTION: Zero-point also added, need to be excluded for optimization
        '''
        #run option        
        if parset != None:
            #run model first with new parameters
            for par in self._get_fitting_parameters().keys():
                self._model.Parameters[par] = parset[par]
        if  self._data.get_measured_times()[0] == 0.:
            self._model._Time = self._data.get_measured_times()
        else:
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
        
        according:  Typically, Q is chosen as the inverse of the measurement 
        error covariance matrix (Marsili–Libelli et al., 2003; 
        Omlin and Reichert, 1999; Vanrolleghem and Dochain, 1998)

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
            self.WSSE += resid * np.linalg.inv(qerr)* resid.transpose()
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
        '''get all model parameters of the current model
        '''
        return self._model.Parameters

    def set_fitting_parameters(self, parlist):
        '''set subset of parameters as basis for optimization
        
        List of parameter names included in the fitting
        or dict + values
        '''
        if isinstance(parlist,list):
            _fitting_pars = {}
            for par in parlist:
                if par in self._model.Parameters:
                    _fitting_pars[par] = self._model.Parameters[par]
                else:
                    raise Exception('Parameter %s is no model parameter.'%par)
        elif isinstance(parlist,dict):               
            _fitting_pars = {}
            for par in parlist:
                if par in self._model.Parameters:
                    _fitting_pars[par] = parlist[par]
                    self._model.Parameters[par] = parlist[par]
                else:
                    raise Exception('Parameter %s is no model parameter.'%par)            
        else:
            raise Exception('List or dictionary is needed!')
        self._fitting_pars = collections.OrderedDict(sorted(_fitting_pars.items(), key=lambda t: t[0]))

    def _get_fitting_parameters(self):
        '''
        '''
        return self._fitting_pars

    
    def local_parameter_optimize(self, initial_parset=None, add_plot=True, method = 'Nelder-Mead', *args, **kwargs):
        '''find parameters for optimal fit
        
        initial_parset: dict!!
        
        method options: Nelder-Mead, 
        
        '''
        #first save the output with the 'old' parameters
        #if initial parameter set given, use this, run and save   
        self._Pre_Optimize = ode_optim_saver()
        if initial_parset != None:
            #control for similarity and update the fitting pars 
            if sorted(initial_parset.keys()) != sorted(self._get_fitting_parameters().keys()):
                print 'Fitting parameters are updated...'
                print 'Previous set of fitting parameters: ',
                print self._get_fitting_parameters().keys()
                print 'New set of fitting parameters: '
                print initial_parset.keys()
            self.set_fitting_parameters(initial_parset)
            
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
        self.optimize_info = optimize.minimize(self.get_WSSE, parray, method= method, *args, **kwargs)
        print self.optimize_info.message
        
        #comparison plot
        if add_plot == True:
            if len(self.Data.columns) == 1:
                fig,axes = plt.subplots(1,1)
                var = self.Data.columns[0]
                #plot data
                axes.plot(self.Data.index, self.Data[var], marker='o', linestyle='none', color='k', label='Measured')
                #plot output old
                axes.plot(self._Pre_Optimize.visual_output.index, self._Pre_Optimize.visual_output[var], linestyle='-.', color='k', label='No optimization')            
                #plot output new
                axes.plot(self._solve_for_visual().index, self._solve_for_visual()[var], linestyle='--', color='k', label='Optimized')
                axes.set_ylabel(var)
                axes.yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
       
                axes.set_xticklabels([])
                # resize for legend
                box1 = axes.get_position()
                axes.set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])
                axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)           
                
                #set time label x-ax
                axes.set_xlabel('Time')
            else:
                fig,axes = plt.subplots(len(self.Data.columns),1)
                fig.subplots_adjust(hspace=0.1)
                for i,var in enumerate(self.Data.columns):
                    #plot data
                    axes[i].plot(self.Data.index, self.Data[var], marker='o', linestyle='none', color='k', label='Measured')
                    #plot output old
                    axes[i].plot(self._Pre_Optimize.visual_output.index, self._Pre_Optimize.visual_output[var], linestyle='-.', color='k', label='No optimization')            
                    #plot output new
                    axes[i].plot(self._solve_for_visual().index, self._solve_for_visual()[var], linestyle='--', color='k', label='Optimized')
                    axes[i].set_ylabel(var)
                    axes[i].yaxis.set_major_locator(MaxNLocator(5, prune='lower'))
       
                axes[0].set_xticklabels([])
                # resize for legend
                box1 = axes[0].get_position()
                axes[0].set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])
                axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)           
                
                #set time label x-ax
                axes[-1].set_xlabel('Time')
            
        return self.optimize_info
        
    def plot_spread_diagram(self, variable, ax = None, marker='o', facecolor='none', 
                             edgecolor = 'k'):
        '''
        Spread_diagram(axs,obs, mod, infobox = True, *args, **kwargs)
                
        '''
        try:
            self.optimize_info
        except:
            raise Exception('Run optimization first!')
        
        if variable not in self._data.get_measured_variables():
            raise Exception('This variable is not listed as measurement')
        
        if ax == None:
            fig,ax = plt.subplots(1,1)

        #prepare dataframe
        varModMeas = pd.concat((self.Data[variable],self.ModelOutput[variable]), axis=1, keys=['Measured','Modelled']) 
        varModMeas = varModMeas.dropna()
        
        ax = Spread_diagram(ax,varModMeas['Measured'], 
                            varModMeas['Modelled'], 
                            infobox = True, marker='o', facecolor='none', 
                            edgecolor = 'k')
        ax.set_xlabel(r'measured')                     
        ax.set_ylabel(r'modelled')
        return ax  
 
        
class ode_FIM(object):
    '''
    OED aimed class functioning
    
    Deliverables:
    - FIM calcluation based on odegenerator model sensitivities
    - link with optimization algorithms for optimization of designs
    
    AIM:
    - Link with robust OED -> sequential design (paropt - exp-opt - paropt)
    '''
    
    def __init__(self, odeoptimizer, sensmethod = 'analytical'): #Measurements
        '''
        Measurements: measurements-class instance
        
        Parameters
        -----------
        sensmethod: analytical|numerical
            analytical is ot always working, but more accurate, since not dependent
            on a selected perturbation factor for sensitivity calcluation              
        
        '''
        if isinstance(odeoptimizer, ode_optimizer):
            self.odeoptimizer = odeoptimizer
        else:
            raise Exception('Input class is ode_optimizer instance!')
        
        self._model = odeoptimizer._model
        self._data = odeoptimizer._data
        self.Error_Covariance_Matrix = odeoptimizer._data._Error_Covariance_Matrix
        self.Parameters = odeoptimizer._model.Parameters
        
        self.criteria_optimality_info = {'A':'min', 'modA': 'max', 'D': 'max', 
                                         'E':'max', 'modE':'min'}
        
        #initially set all measured variables as included
        self.set_variables_for_FIM(self.get_measured_variables())
        
        #Run sensitivity
        if  self._data.get_measured_times()[0] == 0.:
            self._model._Time = self._data.get_measured_times()
        else:
            self._model._Time = np.concatenate((np.array([0.]),self._data.get_measured_times()))
        
#        self._model._Time = np.concatenate((np.array([0.]),self._data.get_measured_times()))
        if sensmethod == 'analytical':
            self._model.analytic_local_sensitivity()
            self.sensitivities = self._model.analytical_sensitivity
        elif sensmethod == 'numerical':
            self._model.numeric_local_sensitivity()
            self.sensitivities = self._model.numerical_sensitivity           
        self._model.set_time(self._model._TimeDict)
        #
        self.get_FIM()        
        
    def get_all_variables(self):
        '''
        returns all model variables in the current model
        '''
        return self._model._model.get_variables()

    def get_measured_variables(self):
        '''
        '''
        if sorted(self._model.get_measured_variables()) != sorted(self._data.get_measured_variables()):
            raise Exception('Model and Data measured variables are not in line with eachother.')
        return self._data.get_measured_variables()
        
    def get_variables_for_FIM(self):
        '''variables for FIM calculation
        '''
        return self._FIMvariables

    def set_variables_for_FIM(self, varlist):
        '''variables for FIM calculation
        '''
        for var in varlist:
            if not var in self.get_measured_variables():
                raise Exception('Variabel %s not measured')
        self._FIMvariables = varlist

    def _get_nonFIM(self):
        '''
        '''
        return [x for x in self.get_measured_variables() if not x in self.get_variables_for_FIM()]

    def get_FIM(self):
        '''  
        TOMORROW!!!! eerst nonFIM dan nans eruit!!
        
        Based on the measurement errors and timesteps to include evaluation,
        the FIM is calculated
        
        calculate e FIM for each timestep and a combine FIM
        
        Notes
        ------
        
        '''        
        
        self.FIM = np.zeros((len(self.Parameters),len(self.Parameters)))
        self.FIM_timestep = {}
        
        Qerr = self.Error_Covariance_Matrix.copy()
        
        check_for_empties=False
        for timestep in self._data.get_measured_times():
            #select subset of Error Covariance based on variables selected for FIM
            for varnot in self._get_nonFIM():
                Qerr[timestep] = Qerr[timestep].drop(varnot,axis=1).drop(varnot)
            
            #control if there is a measured value for this timestep 
            #for all used values (check Error Covariance, since this one is prepared)
            if Qerr[timestep].empty:
                check_for_empties = True
            else:
                #GET SENS_MATRIX
                #varibales in time that are available
                nbvar = Qerr[timestep].shape[0]
                
                sensmatrix = np.zeros((nbvar,len(self.Parameters)))
                #create sensitivity amtrix
                for i,var in enumerate(Qerr[timestep].columns):
                    sensmatrix[i,:] = np.array(self.sensitivities[var].xs(timestep))
    
                #calculate matrices
                FIMt = np.matrix(sensmatrix).transpose() * np.linalg.inv(np.matrix(Qerr[timestep])) * np.matrix(sensmatrix)
                self.FIM = self.FIM + FIMt
                self.FIM_timestep[timestep] = FIMt
        if check_for_empties == True:
            print 'Not all timesteps are evaluated'
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

    
    
    
    