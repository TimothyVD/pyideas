# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:57:52 2015

@author: timothy
"""
from pyideas.model import AlgebraicModel
from pyideas.sensitivity import NumericalLocalSensitivity, DirectLocalSensitivity
from pyideas.confidence import TheoreticalConfidence
from pyideas.uncertainty import Uncertainty
from pyideas.oed import RobustOED
from pyideas.parameterdistribution import ModPar

import numpy as np


parameters = {'W0': 2.0805,
              'Wf': 9.7523,
              'mu': 0.0659}

system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}

M1 = AlgebraicModel('Modsim1', system, parameters, ['t'])

M1.independent = {'t': np.array([0., 20., 29., 41., 50., 65., 72.])}

M1.variables_of_interest = ['W']

M1.initialize_model()

M1.run()

#M1sens = NumericalLocalSensitivity(M1, parameters.keys(), perturbation=1e-6)
M1sens = DirectLocalSensitivity(M1)

M1uncertainty = Uncertainty({'W': '1'})

M1conf = TheoreticalConfidence(M1sens, M1uncertainty)

M1oed = RobustOED(M1conf, independent_samples=5)

M1oed.set_parameter_distributions([ModPar('W0', 0.0, 20.0, 'randomUniform'),
                                   ModPar('Wf', 0.0, 20.0, 'randomUniform'),
                                   ModPar('mu', 0.0, 2.0, 'randomUniform')])

M1oed.set_independent_distributions([ModPar('t', 0.0, 80.0, 'randomUniform')])

par_set = [M1oed._oed['par']._dof_dict_to_array(M1.parameters)]

M1oed.maximin(K_max=1)