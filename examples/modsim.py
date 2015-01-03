# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:45:57 2014

@author: timothy
"""

# general python imports
from __future__ import division
import os
import pandas as pd

# bio-intense custom developments
from biointense import DAErunner, ode_measurements, ode_optimizer, ode_FIM


def run_modsim_models():
    # Data
    file_path = os.path.join(os.getcwd(), 'data', 'grasdata.xls')
    data = pd.read_excel(file_path, 'Blad1', names=['time', 'W'])
    measurements = ode_measurements(data)
    
    # Logistic
    
    Parameters = {'W0': 2.0805,
                  'Wf': 9.7523,
                  'mu': 0.0659}
    
    Alg = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}
    
    M1 = DAErunner(Parameters=Parameters, Algebraic=Alg,
                   Modelname='Modsim1', print_on=False)
    
    M1.set_xdata({'start': 0, 'end': 72, 'nsteps': 1000})
    M1.set_measured_states(['W'])
    
    optim1 = ode_optimizer(M1, measurements, print_on=False)
    optim1.local_parameter_optimize(add_plot=False)
    
    FIM_stuff1 = ode_FIM(optim1, print_on=False)
    FIM_stuff1.get_newFIM()
    FIM_stuff1.get_parameter_confidence()
    FIM_stuff1.get_parameter_correlation()

    # Exponential
    Parameters = {'Wf': 10.7189,
                  'mu': 0.0310}

    Alg = {'W': 'Wf*(1-exp(-mu*t))'}

    M2 = DAErunner(Parameters=Parameters, Algebraic=Alg,
                   Modelname='Modsim2', print_on=False)

    M2.set_initial_conditions({'Dump': 0})
    M2.set_xdata({'start': 0, 'end': 72, 'nsteps': 1000})
    M2.set_measured_states(['W'])

    optim2 = ode_optimizer(M2, measurements, print_on=False)
    optim2.local_parameter_optimize(add_plot=False)

    FIM_stuff2 = ode_FIM(optim2, print_on=False)
    FIM_stuff2.get_newFIM()
    FIM_stuff2.get_parameter_confidence()
    FIM_stuff2.get_parameter_correlation()

    # Gompertz
    Parameters = {'W0': 2.0424,
                  'D': 0.0411,
                  'mu': 0.0669}

    Alg = {'W': 'W0*exp((mu*(1-exp(-D*t)))/(D))'}

    M3 = DAErunner(Parameters=Parameters, Algebraic=Alg,
                   Modelname='Modsim3', print_on=False)

    M3.set_xdata({'start': 0, 'end': 72, 'nsteps': 1000})
    M3.set_measured_states(['W'])

    M3.calcAlgLSA()

    optim3 = ode_optimizer(M3, measurements, print_on=False)
    optim3.local_parameter_optimize(add_plot=False)
    
    FIM_stuff3 = ode_FIM(optim3, print_on=False)
    FIM_stuff3.get_newFIM()
    FIM_stuff3.get_parameter_confidence()
    FIM_stuff3.get_parameter_correlation()

if __name__ == "__main__":
    run_modsim_models()
