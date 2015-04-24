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
    >>> def uncertainty_y1(y1):
    >>>     return (0.1*y1)**2
    >>>
    >>> uncertainty_dict = {'y1': '0.05*y1',
                            'y2': '0.09*y2'}
    >>>
    """

    def __init__(self, uncertainty_dict):
        """
        """
        self.uncertainty_dict = uncertainty_dict

    def get_uncertainty(self, output):
        """
        """
        uncertainty = output.copy()
        for var, value in self.uncertainty_dict.items():
            if isinstance(value, str):
                sympy_uncertainty = sympy.sympify(value)
                fun = sympy.lambdify(var, sympy_uncertainty)
                uncertainty[var] = fun(output[var])
            elif hasattr(value, '__call__'):
                uncertainty[var] = value(output[var])
            else:
                raise Exception('Only strings and functions can be passed!')

        return uncertainty
