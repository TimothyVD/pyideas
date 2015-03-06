# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 19:20:53 2015

@author: timothy
"""
import numpy as np
import pandas as pd
import os

import biointense
from biointense.model import AlgebraicModel
from biointense.optimisation import ParameterOptimisation
from biointense import ModPar


def run_modsim_models_old():

    # Data
    file_path = os.path.join(biointense.BASE_DIR, '..', 'examples', 'data',
                             'grasdata.csv')
    data = pd.read_csv(file_path, header=0, names=['time', 'W'])
    measurements = biointense.ode_measurements(data)

    # Logistic

    Parameters = {'W0': 20.0805,
                  'Wf': 0.97523,
                  'mu': 0.10}

    Alg = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = biointense.DAErunner(Parameters=Parameters, Algebraic=Alg,
                              Modelname='Modsim1', print_on=False)

    M1.set_xdata({'start': 0, 'end': 72, 'nsteps': 1000})
    M1.set_measured_states(['W'])

    optim = biointense.ode_optimizer(M1, measurements, print_on=False)
    return optim.local_parameter_optimize(add_plot=False).x

def run_modsim_models_new():
    # Data
    file_path = os.path.join(biointense.BASE_DIR, '..', 'examples', 'data',
                             'grasdata.csv')
    data = pd.read_csv(file_path, header=0, names=['t', 'W'])
    measurements = biointense.ode_measurements(data, xdata='t')

    parameters = {'W0': 20.0805,
                  'Wf': 0.97523,
                  'mu': 0.10}

    system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = AlgebraicModel('Modsim1', system, parameters)

    M1.set_independent({'t': np.array([0., 4., 6., 41., 50., 65., 72.])})

    M1.set_variables_of_interest(['W'])

    M1.initialize_model()

    M1.run()

    optim = ParameterOptimisation(M1 , measurements, ['W0', 'Wf', 'mu'])

    optim.local_optimize(obj_fun='wsse')

#    optim.set_fitting_par_distributions([ModPar('Wf', 0.0, 20.0,
#                                                'randomUniform'),
#                                         ModPar('mu', 0.0, 20.0,
#                                                'randomUniform')])
#
#    final_pop, ea = optim.bioinspyred_optimize()

    return optim.local_optimize(obj_fun='wsse').x

if __name__ == "__main__":
    optim_old = run_modsim_models_old()
    optim_new = run_modsim_models_new()

    np.testing.assert_allclose(optim_old, optim_new, rtol=1e-5)




