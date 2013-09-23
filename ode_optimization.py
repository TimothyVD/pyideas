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
from parameterdistribution import *

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
        
        self._distributions_set = False

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
        if len(self.Data.columns)>1:
            fig,axes = plt.subplots(len(self.Data.columns),1)
            for i,var in enumerate(self.Data.columns):
                axes[i].plot(self.Data.index, self.Data[var], marker='o', linestyle='none', color='k')
                axes[i].plot(self._solve_for_visual().index, self._solve_for_visual()[var], linestyle='--', color='k')
        else:
            var = self.Data.columns[0]
            fig,axs = plt.subplots(1,1)
            axs.plot(self.Data.index, self.Data[var], marker='o', linestyle='none', color='k')
            axs.plot(self._solve_for_visual().index, self._solve_for_visual()[var], linestyle='--', color='k')            
        
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
        returns the parameters currently set as fitting
        '''
        return self._fitting_pars


    def _pre_optimize_save(self, initial_parset=None):
        """
        """
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
        
        return parray
        
    
    def local_parameter_optimize(self, initial_parset=None, add_plot=True, method = 'Nelder-Mead', *args, **kwargs):
        '''find parameters for optimal fit
        
        initial_parset: dict!!
        
        method options: Nelder-Mead, Powell
        
        '''
        #first save the output with the 'old' parameters
        #if initial parameter set given, use this, run and save   
        parray = self._pre_optimize_save(initial_parset=initial_parset)
        
        #OPTIMIZATION
        #TODO: ADD OPTION FOR SAVING THE PARSETS (IN GETWSSE!!)
        #different algorithms: but implementation  Anneal and CG are not working 
        #a first fix made Powell work
        self.optimize_info = optimize.minimize(self.get_WSSE, parray, method= method, *args, **kwargs)
        print self.optimize_info.message
        
        if add_plot == True:
            self._add_optimize_plot()

        return self.optimize_info

    def set_fitting_par_distributions(self,pardistrlist):
        """
        For each parameter set as fitting parameter, the information
        of the distribution is set.
        
        Parameters
        ------------
        pardistrlist : list
            List of ModPar instances
        optguess : boolean
            Put this True if you want to update the currently saved optimal 
            guess value for the parameters
        
        """
        #Checking if distirbutions are already set
        if not self._distributions_set:
            self.pardistributions={}
            self._distributions_set = True
            
        if isinstance(pardistrlist,ModPar): #one parameter
            if len(self._get_fitting_parameters()) > 1:
                raise Exception("""Only one parameterdistirbution is given, 
                whereas the number of fitting parameters is %d
                """ %len(self._get_fitting_parameters()))
            else:
                if pardistrlist.name in self._fitting_pars:

                        
                    if pardistrlist.name in self.pardistributions:
                        print 'Parameter distribution info updated for %s' %pardistrlist.name
                        self.pardistributions[pardistrlist.name] = pardistrlist
                    else:
                        self.pardistributions[pardistrlist.name] = pardistrlist
                else:
                    raise Exception('Parameter is not listed as fitting parameter')       
        
        elif isinstance(pardistrlist,list):
            #A list of ModPar instances
            for parameter in pardistrlist:
                if parameter.name in self._fitting_pars:
                    if not parameter.min<self._fitting_pars[parameter.name]<parameter.max:
                        raise Exception('Current guess is not between min and max value of the parameter!')
                    if parameter.name in self.pardistributions:
                        print 'Parameter distribution info updated for %s' %parameter.name
                        self.pardistributions[parameter.name] = parameter
                    else:
                        self.pardistributions[parameter.name] = parameter
                else:
                    raise Exception('Parameter %s is not listed as fitting parameter' %parameter.name)                
        else:
            raise Exception("Bad input type, give list of ModPar instances.")

    def get_fitting_par_distributions_from_file(self,parinfofile):
        """
        For each parameter set as fitting parameter, the information
        of the distribution is taken from an ASCII file
        
        Parameters
        ------------
        parinfofile : file
            ASCII file with the parameter information

        Notes
        ------        
        The file need to be setup according to following format, with each
        parameter filling one line:
        
        parametername minvalue maxalue distributionname optarg1 optarg2 ...
        
        """
        parlist=[]
        f = open(parinfofile)
        cnt=1
        for line in f:
            sline = line.strip().split(' ')
            if len(sline) == 4:
                print sline
                par = ModPar(sline[0],float(sline[1]),float(sline[2]),sline[3])
            elif len(sline) == 5:
                par = ModPar(sline[0],float(sline[1]),float(sline[2]),sline[3],float(sline[4]))              
            elif len(sline) == 6:
                print sline
                par = ModPar(sline[0],float(sline[1]),float(sline[2]),sline[3],float(sline[4]),float(sline[5]))
            else:
                raise Exception('Too much arguments on line %d' %cnt)
            parlist.append(par)
            cnt+=1
        f.close()
        
        self.set_fitting_par_distributions(parlist)      


    def bioinspyred_optimize(self, initial_parset=None, add_plot=True):
        """
        
        Notes
        ------
        A working version of Bio_inspyred is needed to get this optimization 
        running!
        """
        from time import time
        from random import Random
        from inspyred import ec
        from inspyred.ec import terminators
        
        #FIRST SAVE THE CURRENT STATE
        parray = self._pre_optimize_save(initial_parset=initial_parset)
        
        #OPTIMIZATION
        
        
        



    def _add_optimize_plot(self):  
        '''quick evaluation plot function
        Quick visualisation of the performed optimization function
              
        
        '''
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
 
        


    
    
    
    