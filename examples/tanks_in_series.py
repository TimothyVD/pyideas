# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:02:35 2014

@author: timothy
"""

from pyideas import Model, 
import numpy as np

#Generate number of tanks
number_of_tanks = 10
system = {}
for i in range(number_of_tanks):
    if i == 0:
        system['dC'+str(i)] = 'Q_in*(C_in - C0)/Vol'
    elif i < number_of_tanks:
        system['dC'+str(i)] = 'Q_in*(C'+str(i-1)+' - C'+str(i)+')/Vol'
        
# Define stepfunction
system['C_in'] = '5/(1+exp(-5*(t-10)))'

# Set parameters
parameters = {'Q_in':5,'Vol':1e2/number_of_tanks}

# Initiate DAErunner object
M1 = Model('TIS', system, parameters)

# Set initial conditions for all tanks
initial_cond= {}
for i in range(number_of_tanks):
    initial_cond['C'+str(i)] = 0.0
M1.initial_conditions = initial_cond

# Set time
M1.independent = {'t': np.linspace(0, 50, 5000)}

# Solve system of ODEs
M1.run().plot()
