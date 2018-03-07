# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:32:53 2015

@author: timothy
"""
from __future__ import division
import numpy as np
import pandas as pd
import os

import pyideas
from pyideas import Model
from pyideas import NumericalLocalSensitivity, DirectLocalSensitivity

def run_fermentor():

    system = {'dS': 'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
              'dX': '-Q_in/V*X+mu_max*S/(S+K_S)*X'}

    parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                  'S_in': 0.02, 'V': 20}

    M_fermentor = Model('fermentor', system, parameters)
    M_fermentor.independent = {'t': np.linspace(0, 100, 5000)}
    M_fermentor.initial_conditions = {'S': 0.02, 'X': 5e-5}

    sens_numeric = NumericalLocalSensitivity(M_fermentor, parameters=['mu_max', 'K_S'])
    sens_analytic = DirectLocalSensitivity(M_fermentor, parameters=['mu_max', 'K_S'])
    #sens_out = sens.get_sensitivity()

    pert_factors = 10.**np.arange(-14, 0, 1)
    out_fig = sens.calc_quality_num_lsa(pert_factors)

    return out_fig, sens_numeric, sens_analytic


if __name__ == "__main__":
    out, sens_numeric, sens_analytic = run_fermentor()
    out['S', 'mu_max'].plot(logx=True, logy=True)
    
    sens_numeric.perturbation = 1e-6
    sens_num_out = sens_numeric.get_sensitivity()['S']
    sens_ana_out = sens_analytic.get_sensitivity()['S']
    sens_num_out.plot()
    sens_ana_out.plot()
    (sens_num_out - sens_ana_out).plot()


