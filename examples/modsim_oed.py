# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 19:20:53 2015

@author: timothy
"""
import numpy as np

from pyideas.model import AlgebraicModel
from pyideas.sensitivity import NumericalLocalSensitivity
from pyideas.confidence import TheoreticalConfidence
from pyideas.oed import BaseOED
from pyideas.parameterdistribution import ModPar
from pyideas import Uncertainty

def run_modsim_models():

    parameters = {'W0': 2.0805,
                  'Wf': 9.7523,
                  'mu': 0.0659}

    system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

    M1 = AlgebraicModel('Modsim1', system, parameters, ['t'])

    M1.independent = {'t': np.array([0., 20., 29., 41., 50., 65., 72.])}

    M1.variables_of_interest = ['W']

    M1.initialize_model()

    M1.run()

    M1sens = NumericalLocalSensitivity(M1)
    M1sens.perturbation = 1e-6
    M1uncertainty = Uncertainty({'W': '1'})
    M1conf = TheoreticalConfidence(M1sens, M1uncertainty)
    M1conf.get_FIM()

    M1oed = BaseOED(M1conf, ['t'])
    M1oed._independent_samples = 7

    M1oed.set_dof_distributions([ModPar('t', 0.0, 80.0, 'randomUniform')])

    final_pop, ea = M1oed.inspyred_optimize()

    M1oed.select_optimal_individual(final_pop)

    return M1conf

if __name__ == "__main__":
    FIM_new = run_modsim_models()

