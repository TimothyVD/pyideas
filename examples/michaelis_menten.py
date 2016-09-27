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

    model = Model('MichaelisMenten', system, parameters)

    model.initial_conditions = {'S': 0.5, 'P': 0.0}

    model.independent = {'t': np.linspace(0, 72, 1000)}
    model.initialize_model()

    return model


def MM_ode(model):
    return model.run(procedure="ode")


def MM_odeint(model):
    return model.run(procedure="odeint")


def MM_odespy(model):
    return model.run(procedure="odespy")

if __name__ == "__main__":
    model = michaelis_menten()
    result1 = MM_ode(model)
    result2 = MM_odeint(model)
