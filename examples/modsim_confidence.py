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
from biointense.sensitivity import NumericalLocalSensitivity
from biointense.confidence import BaseConfidence

def run_modsim_models_old():

    # Data
    file_path = os.path.join(biointense.BASE_DIR, '..', 'examples', 'data',
                             'grasdata.csv')
    data = pd.read_csv(file_path, header=0, names=['time', 'W'])
    measurements = biointense.ode_measurements(data)

    # Logistic

    Parameters = {'W0': 2.0805,
                  'Wf': 9.7523,
                  'mu': 0.0659}

    Alg = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = biointense.DAErunner(Parameters=Parameters, Algebraic=Alg,
                              Modelname='Modsim1', print_on=False)

    M1.set_xdata({'start': 0, 'end': 72, 'nsteps': 1000})
    M1.set_measured_states(['W'])

    optim1 = biointense.ode_optimizer(M1, measurements, print_on=False)
    optim1.local_parameter_optimize(add_plot=False)

    FIM_stuff1 = biointense.ode_FIM(optim1, print_on=False)
    FIM_stuff1.get_newFIM()

    return FIM_stuff1


def run_modsim_models_new():

    parameters = {'W0': 2.0805,
                  'Wf': 9.7523,
                  'mu': 0.0659}

    system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = AlgebraicModel('Modsim1', system, parameters)

    M1.set_independent({'t': np.array([0., 20., 29., 41., 50., 65., 72.])})

    M1.set_variables_of_interest(['W'])

    M1.initialize_model()

    M1.run()

    M1sens = NumericalLocalSensitivity(M1, parameters.keys(), perturbation=1e-6)

    M1conf = BaseConfidence(M1sens)
    M1conf.model = M1

    error = np.zeros([1, len(M1._independent_values['t'])]) + 1
    M1conf.uncertainty_PD = pd.DataFrame(np.concatenate(
        [np.atleast_2d(M1._independent_values['t']),
         error], axis=0).T, columns=['t', 'W']).set_index('t')
    M1conf.FIM

    return M1conf

if __name__ == "__main__":
    FIM_old = run_modsim_models_old()
    FIM_new = run_modsim_models_new()
    # FIXME!
    np.testing.assert_allclose(FIM_old.FIM, FIM_new.FIM, rtol=1e-2)
    np.testing.assert_allclose(FIM_old.get_parameter_confidence(),
                               FIM_new.get_parameter_confidence(), rtol=1e-2)
    np.testing.assert_allclose(FIM_old.get_parameter_correlation(),
                               FIM_new.get_parameter_correlation(), rtol=1e-2)

    np.testing.assert_allclose(FIM_old.get_model_confidence()['W'],
                               FIM_new.get_model_confidence()['W'], rtol=0.02)

#(sens_PD['W']-M1.calcAlgLSA(Sensitivity='CPRS')['W']).plot()
