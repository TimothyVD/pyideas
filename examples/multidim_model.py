# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 15:42:25 2015

@author: timothy
"""

# general python imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# new
from biointense.model import AlgebraicModel
from biointense.sensitivity import NumericalLocalSensitivity

system = {'v': 'Vf*SA*SB/(Kp*SA + Km*SB + SA*SB)'}
parameters = {'Vf': 0.0839, 'Kp': 3.52, 'Km': 143.2}

M1 = AlgebraicModel('Shin_Kim', system, parameters)

SA = np.linspace(0.1, 800., 100)
SB = np.linspace(0.1, 10., 50)

M1.set_independent({'SA': SA, 'SB': SB})

M1.initialize_model()

output = M1.run()

M1.plot_contourf('SB', 'SA', output['v'])

sens = NumericalLocalSensitivity(M1, ['Vf', 'Km', 'Kp'])
output_sens = sens.get_sensitivity()

plt.figure()
independent1 = 'SA'
independent2 = 'SB'
shape = M1._independent_len.values()

x = M1._independent_values[independent1]
x = np.reshape(x, shape)
y = M1._independent_values[independent2]
y = np.reshape(y, shape)
z = output_sens['v', 'Kp'].values
z = np.reshape(z, shape)

plt.contourf(x, y, z)
