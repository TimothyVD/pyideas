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

# new
from pyideas.model import AlgebraicModel
from pyideas.sensitivity import NumericalLocalSensitivity
from pyideas.confidence import TheoreticalConfidence
from pyideas.uncertainty import Uncertainty
from pyideas import Measurements
from pyideas.oed import BaseOED
from pyideas.parameterdistribution import ModPar
from pyideas.optimisation import ParameterOptimisation

# PP = Acetophenone
# PQ = Alanine
system = {'v':  'Vf*SA*SB/(Kp*SA + Km*SB + SA*SB)'}
parameters = {'Vf': 0.0839, 'Kp': 3.52, 'Km': 143.2}
independent = ['SA', 'SB']

M1 = AlgebraicModel('Shin_Kim_forward', system, parameters, independent)

SA = np.linspace(0.1, 800., 1000)
SB = np.linspace(0.1, 10., 100)

cartesian_independent = M1.cartesian({'SA': SA, 'SB': SB})
M1.independent = cartesian_independent

M1.initialize_model()

output = M1.run()

fig, ax = plt.subplots(1,1)
M1.plot_contourf('SA', 'SB', output['v'], ax=ax)

sens = NumericalLocalSensitivity(M1, ['Vf', 'Kp', 'Km'])
output_sens = sens.get_sensitivity(method='PRS')

plt.figure()
M1.plot_contourf('SA', 'SB', output_sens['v', 'Vf'])
plt.title('Local Sensitivity of v to Vf')


def data_from_excel(df, columns, rows, **kwargs):
    df = df[columns].iloc[rows]
    try:
        df.columns = kwargs.get('new_col_names')
    except:
        pass
    try:
        df = df.set_index(kwargs.get('new_index'))
    except:
        pass
    return df

data = pd.read_excel('data/141024_full kinetic data_IPA BA_ATA50.xlsx',
                'All kinetic data', header=None)

new_data = data_from_excel(data,[0,1,2], range(5,55),
                           new_col_names=['SA', 'SB' , 'v'])
new_data = new_data.reset_index(drop=True)
exp_output = new_data['v']

index = pd.MultiIndex.from_arrays([new_data['SA'], new_data['SB']], names=['SA', 'SB'])
exp_output.index = index
measurements = Measurements(pd.DataFrame(exp_output))
measurements.add_measured_errors({'v': 0.101}, method='relative')


optim = ParameterOptimisation(M1, measurements)
optim.local_optimize(obj_crit='wsse')

optim.modmeas.plot()

uncertainty = Uncertainty({'v': '(0.101*v)**2'})
conf = TheoreticalConfidence(sens, uncertainty)

conf.get_FIM()
conf.get_parameter_confidence()

M1oed = BaseOED(conf, ['SA', 'SB'])

M1oed.set_dof_distributions([ModPar('SA', 0.1, 800.0, 'randomUniform'),
                             ModPar('SB', 0.1, 10.0, 'randomUniform')])

   
#final_pop, ea = M1oed.bioinspyred_optimize(pop_size=500, max_eval=3000)
#
#array = M1oed.select_optimal_individual(final_pop).candidate

optim_exp, FIM_total = M1oed.brute_oed({'SA': 800, 'SB': 100}, 50)

M1.independent = optim_exp

conf.get_FIM()
conf.get_parameter_confidence()



