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
from pyideas.model import AlgebraicModel
from pyideas.sensitivity import NumericalLocalSensitivity

system = {'v': 'Vf*SA*SB/(Kp*SA + Km*SB + SA*SB)'}
parameters = {'Vf': 0.0839, 'Kp': 3.52, 'Km': 143.2}
independent = ['SA', 'SB']

M1 = AlgebraicModel('Shin_Kim', system, parameters, independent)

SA = np.linspace(0.01, 800., 1000)
SB = np.linspace(0.01, 10., 100)

cartesian_independent = M1.cartesian({'SA': SA, 'SB': SB})
M1.independent = cartesian_independent

M1.initialize_model()

output = M1.run()

M1.plot_contourf('SA', 'SB', output['v'])

sens = NumericalLocalSensitivity(M1, ['Vf', 'Km', 'Kp'])
output_sens = sens.get_sensitivity()

plt.figure()
plt.contourf(x, y, z)
plt.title('Local sensitivity of v to Kp')
plt.colorbar()