# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 19:20:53 2015

@author: timothy
"""
import numpy as np
import pandas as pd
import os

import pyideas
from pyideas import (AlgebraicModel, ParameterOptimisation, ModPar, Measurements)


def run_modsim_models_new():
    # Data
    file_path = os.path.join(pyideas.BASE_DIR, '..', 'examples', 'data',
                             'grasdata.csv')
    data = pd.read_csv(file_path, header=0, names=['t', 'W'])
    measurements = Measurements(data.set_index('t'))
    measurements.add_measured_errors({'W': 1.}, method='absolute')

    parameters = {'W0': 20.0805,
                  'Wf': 0.97523,
                  'mu': 0.10}

    system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = AlgebraicModel('Modsim1', system, parameters, ['t'])

    M1.independent = {'t': np.array([0., 4., 6., 41., 50., 65., 72.])}

    M1.variables_of_interest = ['W']

    M1.initialize_model()

    M1.run()

    optim = ParameterOptimisation(M1 , measurements,
                                  optim_par=['W0', 'Wf', 'mu'])

#    optim.set_dof_distributions([ModPar('W0', 0.0, 20.0, 'randomUniform'),
#                                 ModPar('Wf', 0.0, 20.0, 'randomUniform'),
#                                 ModPar('mu', 0.0, 2.0, 'randomUniform')])
#
#    final_pop, ea = optim.inspyred_optimize(approach='SA', pop_size=50,
#                                            max_eval=2000)
#    min(final_pop)

    return optim.local_optimize(obj_crit='wsse').x

if __name__ == "__main__":
    optim_new = run_modsim_models_new()



