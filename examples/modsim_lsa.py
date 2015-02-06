# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:32:53 2015

@author: timothy
"""
from __future__ import division
import numpy as np
import pandas as pd
import os

import biointense
from biointense.model import Model
from biointense.sensitivity import NumericalLocalSensitivity

# bio-intense custom developments
from biointense import DAErunner

def run_fermentor_old():

    # Logistic

    ODE = {'dS':'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
           'dX':'-Q_in/V*X+mu_max*S/(S+K_S)*X'}

    Parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                  'S_in': 0.02, 'V': 20}

    M_fermentor = DAErunner(ODE=ODE, Parameters=Parameters,
                            Modelname='Fermentor', print_on=False)

    M_fermentor.set_initial_conditions({'dS':0.02,'dX':5e-5})
    M_fermentor.set_xdata({'start':0,'end':100,'nsteps':5000})
    M_fermentor.set_measured_states(['S','X'])

    pert_factors = 10.**np.arange(-14, 0, 1)
    out_fig = M_fermentor.calc_quality_num_lsa(pert_factors)

    return out_fig, M_fermentor


def run_fermentor_new():

    system = {'dS': 'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
              'dX': '-Q_in/V*X+mu_max*S/(S+K_S)*X'}

    parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                  'S_in': 0.02, 'V': 20}

    M_fermentor = Model('fermentor', system, parameters)
    M_fermentor.set_independent('t', np.linspace(0, 100, 1000))
    M_fermentor.set_initial({'S': 0.02, 'X': 5e-5})

    sens = NumericalLocalSensitivity(M_fermentor, ['mu_max', 'K_S'])
    sens_out = sens.get_sensitivity()

    sens.get_sensitivity_accuracy(criterion='SSE')
    pert_factors = 10.**np.arange(-14, 0, 1)
    out_fig = sens.calc_quality_num_lsa(['mu_max', 'K_S'], pert_factors)

    return out_fig, sens


if __name__ == "__main__":
    out_old, M_old = run_fermentor_old()
    out_old['S', 'mu_max'].plot(logx=True, logy=True)
    out_new, sens = run_fermentor_new()
    out_new['S', 'mu_max'].plot(logx=True, logy=True)
