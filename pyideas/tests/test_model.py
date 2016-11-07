# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 16:30:34 2015

@author: joris
"""
from __future__ import division

import unittest

import numpy as np

from pyideas.model import Model


class TestAlgebraicModel(unittest.TestCase):

    def test_model_instantiate(self):

        system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}
        parameters = {'W0': 2.0805,
                      'Wf': 9.7523,
                      'mu': 0.0659}

        model = Model('Modsim1', system, parameters)

        model.independent = {'t': np.linspace(0, 72, 1000)}

        expected = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}
        result = model.systemfunctions['algebraic']
        assert result == expected

        assert model._ordered_var['algebraic'] == ['W']

    def test_def_creation(self):

        system = {'W': 'W0*Wf/(W0+(Wf-W0)*exp(-mu*t))'}
        parameters = {'W0': 2.0805,
                      'Wf': 9.7523,
                      'mu': 0.0659}

        model = Model('Modsim1', system, parameters)
        model.independent = {'t': np.linspace(0, 72, 1000)}
        model.initialize_model()

        # str version check
        algref = ("def fun_alg(independent, parameters, *args, **kwargs):\n"
                  "    t = independent['t']\n\n"
                  "    W0 = parameters['W0']\n"
                  "    Wf = parameters['Wf']\n"
                  "    mu = parameters['mu']\n\n"
                  "    W = W0*Wf/(W0+(Wf-W0)*np.exp(-mu*t))"
                  " + np.zeros(len(t))\n\n"
                  "    nonder = np.array([W]).T\n"
                  "    return nonder")
        assert algref == model.fun_alg_str

        result = model.fun_alg(model._independent_values, parameters)

        # def calc check
        assert result[-1] == 9.4492688322077534


class TestOdeModel(unittest.TestCase):

    def test_model_instantiate(self):

        ODE = {'dS': 'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
               'dX': '-Q_in/V*X+mu_max*S/(S+K_S)*X'}

        parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                      'S_in': 0.02, 'V': 20}

        model = Model('Fermentor', ODE, parameters)

        # system parsing
        result = model.systemfunctions['ode']
        assert result == {'S': ODE['dS'], 'X': ODE['dX']}

        assert model._ordered_var['ode'] == ['S', 'X']

    def test_def_creation(self):

        ODE = {'dS': 'Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X',
               'dX': '-Q_in/V*X+mu_max*S/(S+K_S)*X',
               'P': 'Q_in*t/X'}

        parameters = {'mu_max': 0.4, 'K_S': 0.015, 'Q_in': 2, 'Ys': 0.67,
                      'S_in': 0.02, 'V': 20}

        model = Model('Fermentor', ODE, parameters)
        model.independent = {'t': [0.02, 5e-5]}
        model.initialize_model()

        # str version check
        algref = ("def fun_alg(independent, parameters, *args, **kwargs):\n"
                  "    t = independent['t']\n\n"
                  "    K_S = parameters['K_S']\n"
                  "    Q_in = parameters['Q_in']\n"
                  "    S_in = parameters['S_in']\n"
                  "    V = parameters['V']\n"
                  "    Ys = parameters['Ys']\n"
                  "    mu_max = parameters['mu_max']\n\n"
                  "    solved_variables = kwargs.get('ode_values')\n"
                  "    S = solved_variables[:, 0]\n"
                  "    X = solved_variables[:, 1]\n\n"
                  "    P = Q_in*t/X + np.zeros(len(t))\n\n"
                  "    nonder = np.array([P]).T\n"
                  "    return nonder")
        assert algref == model.fun_alg_str

        oderef = ("def fun_ode(odes, t, parameters, *args, **kwargs):\n"
                  "    K_S = parameters['K_S']\n"
                  "    Q_in = parameters['Q_in']\n"
                  "    S_in = parameters['S_in']\n"
                  "    V = parameters['V']\n"
                  "    Ys = parameters['Ys']\n"
                  "    mu_max = parameters['mu_max']\n\n"
                  "    S = odes[0]\n"
                  "    X = odes[1]\n\n"
                  "    P = Q_in*t/X\n\n"
                  "    dS = Q_in/V*(S_in-S)-1/Ys*mu_max*S/(S+K_S)*X\n"
                  "    dX = -Q_in/V*X+mu_max*S/(S+K_S)*X\n"
                  "    return [dS, dX]")
        assert oderef == model.fun_ode_str

        result = model.fun_ode(model._independent_values['t'], np.nan,
                               parameters)

        np.testing.assert_almost_equal(-1.7057569296375266e-05, result[0],
                                       decimal=14)
        np.testing.assert_almost_equal(6.428571428571429e-06, result[1],
                                       decimal=14)
