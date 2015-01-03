# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:45:57 2014

@author: timothy
"""

#general python imports
from __future__ import division
import pandas as pd
import os
from collections import OrderedDict

#bio-intense custom developments
from biointense import *

def fermentor_FIM():
    # Read data bioreactor
    file_path = os.path.join(os.getcwd(), 'data', 'fermentor_data.xls')
    data = pd.read_excel(file_path, 'Blad1', names=['time','S','X'])
    measurements = ode_measurements(data, print_on=False)
    
    ODE = {'dS':'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
           'dX':'-Q_in/V*X+mu_max*S/(S+K_S)*X'}
    
    Parameters = {'mu_max':0.4,'K_S':0.015,'Q_in':2,'Ys':0.67,'S_in':0.02,'V':20}
     
    M_fermentor = DAErunner(ODE=ODE, Parameters=Parameters, 
                            Modelname='Fermentor', print_on=False)
    
    M_fermentor.set_initial_conditions({'dS':0.02,'dX':5e-5})
    M_fermentor.set_xdata({'start':0,'end':100,'nsteps':5000})
    M_fermentor.set_measured_states(['S','X'])
    
    # Solve model
    M_fermentor.solve_ode(plotit=False)
    
    optim = ode_optimizer(M_fermentor, measurements, print_on=False)
    optim.set_fitting_parameters(OrderedDict({'mu_max':0.4,'K_S':0.015}))
    optim.local_parameter_optimize()
    
    FIM_stuff = ode_FIM(optim, print_on=False)
    FIM_stuff.get_newFIM()
    FIM_stuff.get_parameter_confidence()
    FIM_stuff.get_parameter_correlation()
    
    print('FIM = ')
    print(FIM_stuff.FIM)
    print('ECM = ')
    print(FIM_stuff.ECM)

if __name__ == "__main__":
    fermentor_FIM()
