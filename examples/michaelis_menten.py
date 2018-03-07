# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 17:32:29 2015

@author: timothy
"""
from __future__ import division

import numpy as np
from pyideas import Model


def michaelis_menten():

    system = {'v': 'Vmax*S/(Ks + S)',
              'dS': '-v',
              'dP': 'v'}
    
    parameters = {'Vmax': 1e-1, 'Ks': 0.5}

    MMmodel = Model('MichaelisMenten', system, parameters)

    MMmodel.initial_conditions = {'S': 0.5, 'P': 0.0}

    MMmodel.independent = {'t': np.linspace(0, 72, 1000)}
    MMmodel.initialize_model()

    return MMmodel


def MM_ode(model):
    return model.run(procedure="ode")


def MM_odeint(model):
    return model.run(procedure="odeint")


def MM_odespy(model):
    return model.run(procedure="odespy")

if __name__ == "__main__":
    MMmodel = michaelis_menten()
    result1 = MM_ode(MMmodel)
    result2 = MM_odeint(MMmodel)
    
    np.testing.assert_allclose(result1, result2, rtol=1e-2)
