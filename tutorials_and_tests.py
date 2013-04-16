# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 14:07:51 2013

@author: VHOEYS
"""

#general python imports
from __future__ import division
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import colors

#bio-intense custom developments
from plotfunctions import *
from ode_generator import odegenerator

#------------------------------------------------------------------------------
#EXAMPLE MODEL
#------------------------------------------------------------------------------
Parameters = {'k1':1/10,'k1m':1/20,
              'k2':1/20,'k2m':1/20,
              'k3':1/200,'k3m':1/175,
              'k4':1/200,'k4m':1/165}
              
System =    {'dEn':'k1m*Es*PP + k4*EP + k2*Es*SB - k1*En*SA - k4*En*PP - k2m*En*PQ',
             'dEs':'- k1m*Es*PP + k3*EsQ - k2*Es*SB + k1*En*SA - k3*Es + k2m*En*PQ',
             'dSA':'- k1*En*SA + k1m*Es*PP',
             'dSB':'- k2*Es*SB + k2m*En*PQ',
             'dPP':'k1*En*SA - k1m*Es*PP - k4*En*PP + k4m*EP',
             'dPQ':'k2*En*SB - k2m*En*PQ - k3*Es*PQ + k3m*EsQ',
             'dEsQ':'k3*Es*PQ - k3m*EsQ',
             'dEP':'k4*En*PP - k4m*EP'}
                        
Modelname = 'MODEL_Halfreaction'

##INITIATE MODEL
M1 = odegenerator(System, Parameters, Modelname = Modelname)
M1.set_measured_states(['SA', 'SB', 'PP', 'PQ'])
M1.set_initial_conditions({'SA':5.,'SB':0.,'En':1.,'EP':0.,'Es':0.,'EsQ':0.,'PP':0.,'PQ':0.})
#M1.set_initial_conditions({'SA':5.,'SB':4.,'En':1.,'EP':6.,'Es':2.5,'EsQ':1.,'PP':1.5,'PQ':0.})
M1.set_time({'start':0,'end':20,'nsteps':1000})

#M1.write_model_to_file(with_sens=False)
#------------------------------------------------------------------------------
#EXAMPLE SET FOR SEMINAR
#------------------------------------------------------------------------------
##run the model
modeloutput = M1.solve_ode(plotit=False)
print modeloutput
#modeloutput.plot(subplots=True) 
##run the taylor approach for identifiability
#M1.taylor_series_approach(2)
##plot Taylor-output
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#M1.plot_taylor_ghost(ax1)
##numerical local sensitivity analysis
#numsens = M1.numeric_local_sensitivity(perturbation_factor=0.0001)
##numerical local sensitivity plot for PP
#numsens['PP'].plot()
##visual check collinearity
#M1.visual_check_collinearity('SA')

#------------------------------------------------------------------------------
#TAYLOR PLOT EXAMPLE FIRST AND SECOND ORDER
#------------------------------------------------------------------------------
#fig = plt.figure()
#fig.subplots_adjust(hspace=0.3)
#ax1 = fig.add_subplot(211)
#ax1 = M1.plot_taylor_ghost(ax1, order = 0, redgreen=True)
##ax1.set_title('First order derivative')
#ax2 = fig.add_subplot(212)
#ax2 = M1.plot_taylor_ghost(ax2, order = 1, redgreen=True)
#ax2.set_title('Second order derivative')

#------------------------------------------------------------------------------
#TEST VAN DE OPSLAGPROBLEMEN
#------------------------------------------------------------------------------
#numerical_sens = {}
#dummy = np.empty((self._Time.size,len(self.Parameters)))
#for key in self._Variables:
#    numerical_sens[key] = pd.DataFrame(dummy, columns = self.Parameters.keys())
#
#for parameter in self.Parameters:
#    print parameter
#
##for parameter in self.Parameters:
#parameter='k1'
#value2save = self.Parameters[parameter]
#print 'sensitivity for parameter ', parameter
##run model with parameter value plus perturbation 
#self.Parameters[parameter] = value2save + perturbation_factor*value2save
#modout_plus = self.solve_ode(plotit = False)
##            modout_plus = pd.DataFrame(modout, columns = self._Variables)
##run model with parameter value minus perturbation 
#self.Parameters[parameter] = value2save - perturbation_factor*value2save
#modout_min = self.solve_ode(plotit = False)        
##            modout_min = pd.DataFrame(modout, columns = self._Variables)
#
##calculate sensitivity for this parameter, all outputs    
##sensitivity indices:
#CAS = (modout_plus-modout_min)/(2.*perturbation_factor*value2save) #dy/dp         
#
##we use now CPRS, but later on we'll adapt to CTRS
#CPRS = CAS*value2save    
##            average_out = (modout_plus+modout_min)/2.
##            CTRS = CAS*value2save/average_out
#
##put on the rigth spot in the dictionary
#for var in self._Variables:
#    print var
#    print parameter
#    numerical_sens[var][parameter] = CPRS[var][:]
#    numerical_sens[var].plot()
#    
##put back original value
#self.Parameters[parameter] = value2save

