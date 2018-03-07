# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 19:20:53 2015

@author: timothy
"""
import numpy as np
import pandas as pd
import os

import pyideas
from pyideas import (AlgebraicModel, NumericalLocalSensitivity,
                     CalibratedConfidence, Measurements, ParameterOptimisation)

def run_modsim_models():
    
    # Define data path of csv file
    file_path = os.path.join(pyideas.BASE_DIR, '..', 'examples', 'data',
                             'grasdata.csv')
    # Data always need to be provided as pd dataframe
    data = pd.read_csv(file_path, header=0, names=['t', 'W'])
    # Data index should contain values for ALL independent variables
    data.set_index('t', inplace=True)
    # Initialise measurement object and define absolute error of 1 for W
    measurements = Measurements(data)
    measurements.add_measured_errors({'W': 1.}, method='absolute')

    # Initialise parameters, real values will be retrieved by optimisation
    parameters = {'W0': 1.,
                  'Wf': 1.,
                  'mu': 1.}

    system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = AlgebraicModel('Modsim', system, parameters, ['t'])

    M1.independent = {'t': np.linspace(0, 72, 1000)}

    M1.initialize_model()
    
    # ParameterOptimisation uses WSSE to reduce offset
    M1optim = ParameterOptimisation(M1, measurements)
    M1optim.local_optimize()

    # Calculate uncertainties for estimated parameter values using FIM
    M1conf = CalibratedConfidence(M1optim)

    return M1, M1conf

if __name__ == "__main__":
    M1, M1conf = run_modsim_models()
    
    # Test whether optimisation has found optimal parameter values
    np.testing.assert_allclose(M1.parameters['W0'], 2.0805, rtol=1e-2)
    np.testing.assert_allclose(M1.parameters['Wf'], 9.7523, rtol=1e-2)
    np.testing.assert_allclose(M1.parameters['mu'], 0.0659, rtol=1e-2)

