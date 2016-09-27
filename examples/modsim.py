# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 16:45:57 2014

@author: timothy
"""

# general python imports
from __future__ import division
import os
import pandas as pd
import numpy as np

import pyideas
from pyideas import (Model, ParameterOptimisation, CalibratedConfidence,
                     Measurements)


def run_modsim_models():

    # Data
    file_path = os.path.join(pyideas.BASE_DIR, '..', 'examples', 'data',
                             'grasdata.csv')
    data = pd.read_csv(file_path, header=0, names=['t', 'W'])
    data = data.set_index('t')
    measurements = Measurements(data)
    measurements.add_measured_errors({'W': 1.0}, method='absolute')

    # Logistic
    parameters = {'W0': 2.0805,
                  'Wf': 9.7523,
                  'mu': 0.0659}

    system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = Model('Modsim1', system, parameters)

    M1.independent = {'t': np.linspace(0, 72, 1000)}

    M1.initialize_model()

    # Perform parameter estimation
    M1optim = ParameterOptimisation(M1, measurements)
    M1optim.local_optimize()

    # Calc parameter uncertainty
    M1conf = CalibratedConfidence(M1optim)
    FIM1 = M1conf.get_FIM()

    # Exponential
    parameters = {'Wf': 10.7189,
                  'mu': 0.0310}

    system = {'W': 'Wf*(1-exp(-mu*t))'}

    M2 = Model('Modsim2', system, parameters)
    M2.independent = {'t': np.linspace(0, 72, 1000)}

    # Perform parameter estimation
    M2optim = ParameterOptimisation(M2, measurements)
    M2optim.local_optimize()

    # Calc parameter uncertainty
    M2conf = CalibratedConfidence(M2optim)
    FIM2 = M2conf.get_FIM()

    # Gompertz
    parameters = {'W0': 2.0424,
                  'D': 0.0411,
                  'mu': 0.0669}

    system = {'W': 'W0*exp((mu*(1-exp(-D*t)))/(D))'}

    M3 = Model('Modsim3', system, parameters)
    M3.independent = {'t': np.linspace(0, 72, 1000)}

    # Perform parameter estimation
    M3optim = ParameterOptimisation(M3, measurements)
    M3optim.local_optimize()

    # Calc parameter uncertainty
    M3conf = CalibratedConfidence(M3optim)
    FIM3 = M3conf.get_FIM()

    return M1, M2, M3, M1conf, M2conf, M3conf


if __name__ == "__main__":
    M1, M2, M3, M1conf, M2conf, M3conf = run_modsim_models()
