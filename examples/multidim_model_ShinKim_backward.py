# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:42:25 2015

@author: timothy
"""

# general python imports
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

# new
import pyideas
from pyideas.model import AlgebraicModel
from pyideas.sensitivity import NumericalLocalSensitivity
from pyideas.confidence import TheoreticalConfidence
from pyideas.uncertainty import Uncertainty
from pyideas.oed import BaseOED
from pyideas.parameterdistribution import ModPar

# Solubility in water
# ----------------------
# SA = MBA: Very soluble
# SB = Pyruvate: 910 mM
# PP = Acetophenone: 46 mM
# PQ = Alanine: 1880 mM

system = {'v': 'Vr*PP*PQ/(Kal*PP + Kac*PQ + PP*PQ + Kal/Kacs*PP**2 + Kac/Kas*PQ**2)'}
parameters = {'Vr': 5.18e-4, 'Kal': 1.07, 'Kac': 0.54, 'Kacs': 1.24, 'Kas': 25.82}
independent = ['PP', 'PQ']

M1 = AlgebraicModel('Shin_Kim_backward', system, parameters, independent)

PP = np.linspace(0.1, 46., 100)
PQ = np.linspace(0.1, 1000., 100)

PP_article = np.array([1., 2., 3., 5., 10.])
PQ_article = np.array([10., 20., 30., 40., 50.])

cartesian_independent = M1.cartesian({'PP': PP_article, 'PQ': PQ_article})
M1.independent = cartesian_independent

M1.initialize_model()

output = M1.run()

fig, ax = plt.subplots(1,1)
M1.plot_contourf('PP', 'PQ', output['v'], ax=ax)

sens = NumericalLocalSensitivity(M1)
#output_sens = sens.get_sensitivity(method='CPRS')

#plt.figure()
#independent1 = 'PP'
#independent2 = 'PQ'
#shape = M1._independent_len.values()
#
#x = M1._independent_values[independent1]
#x = np.reshape(x, shape)
#y = M1._independent_values[independent2]
#y = np.reshape(y, shape)
#z = output_sens['v', 'Vr'].values
#z = np.reshape(z, shape)
#
#plt.contourf(x, y, z)
#plt.colorbar()


uncertainty = Uncertainty({'v': '(5e-2*v)**2'})
conf = TheoreticalConfidence(sens, uncertainty)

conf.get_parameter_confidence()

M1oed = BaseOED(conf, ['PP', 'PQ'])

M1oed.set_dof_distributions([ModPar('PP', 0.1, 10.0, 'randomUniform'),
                             ModPar('PQ', 0.1, 50.0, 'randomUniform')])

#final_pop, ea = M1oed.bioinspyred_optimize(pop_size=500, max_eval=3000)
#
#array = M1oed.select_optimal_individual(final_pop).candidate

optim_exp, FIM_total = M1oed.brute_oed({'PP': 40, 'PQ': 100}, 25)

M1.independent = optim_exp

par_conf_new = conf.get_parameter_confidence()
