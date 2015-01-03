# -*- coding: utf-8 -*-
"""
Created on Fri May 16 12:02:35 2014

@author: timothy
"""

from biointense import DAErunner
import numpy as np

#Generate number of tanks
number_of_tanks = 10
System = {}
for i in range(number_of_tanks):
    if i == 0:
        System['dC'+str(i)] = 'Q_in*(C_in - C0)/Vol'
    elif i < number_of_tanks:
        System['dC'+str(i)] = 'Q_in*(C'+str(i-1)+' - C'+str(i)+')/Vol'

# Set parameters
Parameters = {'Q_in':5,'Vol':1e1/number_of_tanks}

# Initiate DAErunner object
M1 = DAErunner(ODE = System, Parameters = Parameters, Modelname = 'TIS',
               external_par=['C_in'])

# Make stepfunction
pulse = M1.makeStepFunction({'C_in':np.array([[0.0,0.0],[5.0,10.0],[30.0,1.5]])})
# Add stepfunction to model
M1.addExternalFunction(pulse)

# Set initial conditions for all tanks
initial_cond= {}
for i in range(number_of_tanks):
    initial_cond['C'+str(i)] = 0.0
M1.set_initial_conditions(initial_cond)

# Set time
M1.set_xdata({'start':0,'end':50,'nsteps':5000})

# Solve system of ODEs
M1.solve_ode()
