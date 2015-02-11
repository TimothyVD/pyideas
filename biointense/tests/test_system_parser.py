# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 22:36:12 2015

@author: stvhoey
"""
from __future__ import division

import numpy as np

from biointense.model import Model
from biointense.modeldefinition import generate_ode_derivative_definition, generate_non_derivative_part_definition

ODE = {'dS' : 'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
        'dX' : '-Q_in/V*X+mu_max*S/(S+K_S)*X',
        'P' : 'Q_in*X'}
        
pars = {'mu_max' : 0.4, 'K_S' : 0.015, 'Q_in' : 2, 'Ys' : 0.67, 
        'S_in' : 0.02,'V' : 20}
              
M_fermentor = Model("fermentor", ODE,  pars)         
M_fermentor.set_independent('t', np.empty(10))
modelstr = generate_ode_derivative_definition(M_fermentor)
algstr = generate_non_derivative_part_definition(M_fermentor)

def test_alg_writepart():
    algref = "def fun_alg(t, parameters, *args, **kwargs):\n    K_S = parameters['K_S']\n    mu_max = parameters['mu_max']\n    Q_in = parameters['Q_in']\n    V = parameters['V']\n    Ys = parameters['Ys']\n    S_in = parameters['S_in']\n\n    solved_variables = args[0]\n    S = solved_variables[:, 0]\n    X = solved_variables[:, 1]\n\n    P = Q_in*X + np.zeros(len(t))\n\n    nonder = np.array([P]).T\n    return nonder"
    assert algref == algstr

def test_der_writepart():
    modelref = "def fun_ode(odes, t, parameters, *args, **kwargs):\n    K_S = parameters['K_S']\n    mu_max = parameters['mu_max']\n    Q_in = parameters['Q_in']\n    V = parameters['V']\n    Ys = parameters['Ys']\n    S_in = parameters['S_in']\n\n    S = odes[0]\n    X = odes[1]\n\n    P = Q_in*X\n\n    dX = -Q_in/V*X+mu_max*S/(S+K_S)*X\n    dS = Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X\n    return [dS, dX]\n\n"
    assert modelref == modelstr
    

