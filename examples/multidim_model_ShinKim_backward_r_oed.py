# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:57:52 2015

@author: timothy
"""
from pyideas.model import AlgebraicModel
from pyideas.sensitivity import NumericalLocalSensitivity, DirectLocalSensitivity
from pyideas.confidence import TheoreticalConfidence
from pyideas.uncertainty import Uncertainty
from pyideas.oed import BaseOED, RobustOED
from pyideas.parameterdistribution import ModPar

import numpy as np
import pandas as pd


system = {'v': 'Vr*PP*PQ/(Kal*PP + Kac*PQ + PP*PQ + Kal/Kacs*PP**2 + Kac/Kas*PQ**2)'}
parameters = {'Vr': 5.18e-4, 'Kal': 1.07, 'Kac': 0.54, 'Kacs': 1.24, 'Kas': 25.82}

M1 = AlgebraicModel('Shin_Kim_backward', system, parameters, ['PP', 'PQ'])

PP = np.linspace(0.1, 46., 100)
PQ = np.linspace(0.1, 1000., 100)

PP_article = np.array([1., 2., 3., 5., 10.])
PQ_article = np.array([10., 20., 30., 40., 50.])

independent_cartesian = M1.cartesian({'PP': PP_article, 'PQ': PQ_article})
M1.independent = independent_cartesian

M1.initialize_model()

#M1sens = NumericalLocalSensitivity(M1)
M1sens = DirectLocalSensitivity(M1)

M1uncertainty = Uncertainty({'v': '(5e-2*v)**2'})

M1conf = TheoreticalConfidence(M1sens, M1uncertainty)

M1bruteoed = BaseOED(M1conf, ['PP', 'PQ'])
M1bruteoed.set_dof_distributions([ModPar('PP', 0.5, 46.0, 'randomUniform'),
                                  ModPar('PQ', 0.5, 1000.0, 'randomUniform')])
num_of_samples = 8
M1bruteoed.brute_oed({'PP': 20, 'PQ': 50}, num_of_samples)


M1oed = RobustOED(M1conf, independent_samples=5)

M1oed.set_parameter_distributions([ModPar('Vr', 1e-7, 1e-1, 'randomUniform'),
                                   ModPar('Kal', 0.01, 10, 'randomUniform'),
                                   ModPar('Kac', 0.01, 10, 'randomUniform'),
                                   ModPar('Kacs', 0.05, 10, 'randomUniform'),
                                   ModPar('Kas', 5.0, 50., 'randomUniform')])

M1oed.set_independent_distributions([ModPar('PP', 0.5, 46.0, 'randomUniform'),
                                     ModPar('PQ', 0.5, 1000.0, 'randomUniform')])
# Takes about 1min
opt_independent, par_sets = M1oed.maximin(K_max=15)

out = M1oed._oed['ind']._dof_array_to_dict(opt_independent)
pd.DataFrame(out['independent'])