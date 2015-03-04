# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 16:59:56 2015

@author: timothy
"""
import numpy as np
import pandas as pd
import sympy


class Uncertainty(object):
    """
    The aim of the uncertainty class is to ease the way to pass uncertainties,
    without reducing the flexibility.

    Example
    -------
    >>>
    >>> uncertainty_dict = {'y1': '0.05*y1',
                            'y2': '0.09*y2'}
    """

    def __init__(self, uncertainty_dict):
        """
        """
        self.uncertainty_dict = uncertainty_dict

    def get_uncertainty(self, modeloutput):
        """
        """
        uncertainty = modeloutput.copy()
        for var in self.uncertainty_dict.keys():
            sympy_uncertainty = sympy.sympify(self.uncertainty_dict[var])
            fun = sympy.lambdify(var, sympy_uncertainty)
            uncertainty[var] = fun(modeloutput[var])

        return uncertainty
