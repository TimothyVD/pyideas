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

# new
from biointense.model import Model


def fermentor_old():
    # Read data bioreactor
    file_path = os.path.join(BASE_DIR, '..', 'examples', 'data',
                             'fermentor_data.csv')
    data = pd.read_csv(file_path, header=0, names=['time', 'S', 'X'])
    measurements = ode_measurements(data, print_on=False)

    ODE = {'dS':'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
           'dX':'-Q_in/V*X+mu_max*S/(S+K_S)*X'}

    Parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                  'S_in': 0.02, 'V': 20}

    M_fermentor = DAErunner(ODE=ODE, Parameters=Parameters,
                            Modelname='Fermentor', print_on=False)

    M_fermentor.set_initial_conditions({'dS':0.02,'dX':5e-5})
    M_fermentor.set_xdata({'start':0,'end':100,'nsteps':5000})
    M_fermentor.set_measured_states(['S','X'])

    # Solve model
    M_fermentor.solve_ode(plotit=True)

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

    return M_fermentor, FIM_stuff


def fermentor_new():
    # Read data bioreactor
    file_path = os.path.join(BASE_DIR, '..', 'examples', 'data',
                             'fermentor_data.csv')
    data = pd.read_csv(file_path, header=0, names=['time', 'S', 'X'])
    #measurements = ode_measurements(data, print_on=False)

    system = {'dS': 'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
              'dX': '-Q_in/V*X+mu_max*S/(S+K_S)*X'}

    parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                  'S_in': 0.02, 'V': 20}

    M_fermentor = Model('Fermentor', system, parameters)
    M_fermentor.set_independent('t', np.linspace(0, 100, 1000))
    M_fermentor.set_initial({'S': 0.02, 'X': 5e-5})

    M_fermentor.initialize_model()

    output = M_fermentor.run()
    output.plot(subplots=True)

    #M_fermentor.set_initial_conditions({'dS':0.02,'dX':5e-5})
    #M_fermentor.set_xdata({'start':0,'end':100,'nsteps':5000})
    #M_fermentor.set_measured_states(['S','X'])

    # Solve model
    #M_fermentor.solve_ode(plotit=True)

    # optim = ode_optimizer(M_fermentor, measurements, print_on=False)
    # optim.set_fitting_parameters(OrderedDict({'mu_max':0.4,'K_S':0.015}))
    # optim.local_parameter_optimize()

    # FIM_stuff = ode_FIM(optim, print_on=False)
    # FIM_stuff.get_newFIM()
    # FIM_stuff.get_parameter_confidence()
    # FIM_stuff.get_parameter_correlation()

    # print('FIM = ')
    # print(FIM_stuff.FIM)
    # print('ECM = ')
    # print(FIM_stuff.ECM)

    return M_fermentor#, FIM_stuff


if __name__ == "__main__":
    M_fermentor, FIM = fermentor_old()
    M_fermentor2 = fermentor_new()
