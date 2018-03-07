# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:27:24 2015

@author: timothy
"""
import numpy as np
import pandas as pd
from pyideas import Model


def run_second_order():
    '''
    y'' + 2*tau*omega*y' + omega**2*y = omega**2*K*u

    x1  = y
    x1' = y' = x2
    x2' = y''= -2*tau*omega*x2 - omega**2*x1 + K*omega**2
    '''
    parameters = {'K': 0.01, 'tau': 0.3, 'omega': 0.6}  # mM

    system = {'dx1': 'x2',
              'dx2': '-2*tau*omega*x2 - (omega**2)*x1 + K*omega**2'}

    M1 = Model('second_order', system, parameters)
    M1.initial_conditions = {'x1': 0, 'x2': 0}
    M1.independent = {'t': np.linspace(0, 20, 10000)}
    M1.initialize_model()

    return M1.run()['x1'].values

if __name__ == "__main__":
    M1 = run_second_order()
