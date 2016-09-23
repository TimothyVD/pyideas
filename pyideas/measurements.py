# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:33:14 2013

@author: Timothy Van Daele; Stijn Van Hoey

0.1 Class version of the ODE generator linkfile for optimization
"""

from __future__ import division
import math

import numpy as np
import pandas as pd

from biointense.uncertainty import Uncertainty

ERROR_FUN = {'absolute': '({})**2',
             'relative': '(({})*({}))**2',
             'Ternbach': '({0}*{1}*(1.+ 1./(({0}/{2})**2 + ({0}/{2}))))**2',
             'direct': '{}'}


class Measurements(object):
    '''class to include measured data in the framework


    Input possibilities (cfr. FME package R):

    ATTENTION: The 'variables' and 'values' is required to let it work!

    '''

    def __init__(self, measdata):
        '''
        '''
        if not isinstance(measdata, pd.DataFrame):
            raise Exception('Measured Data must be added as pandas DataFrame')

        self._calculated = False

        self._data = measdata.values.astype(np.float64)
        self._variables = measdata.columns.tolist()

        self._independent_names = measdata.index.names
        indep_values = np.atleast_2d(measdata.index.values.tolist())
        self._independent_values = dict(zip(self._independent_names,
                                            indep_values.astype(np.float64)))

        self._uncertainty = None
        self._meas_uncertainty = None

    def _calc_meas_uncertainty(self):
        """
        """
        ecm = self._uncertainty._get_uncertainty(self._data, self._variables)

        #ecm *= self._data.applymap(lambda x: np.nan if math.isnan(x) else 1)
        #ecm = ecm.applymap(lambda x: 1e50 if math.isnan(x) else x)
        self._meas_uncertainty = ecm
        self._calculated = True

    @property
    def meas_uncertainty(self):
        """
        """
        if self._uncertainty is None:
            raise Exception('First define measurement errrors!')
        if not self._calculated:
            self._calc_meas_uncertainty()
        return self._meas_uncertainty

    def add_measured_errors(self, meas_error_dict, method='relative',
                            lower_accuracy_bound=None,
                            minimal_relative_error=None):
        r'''calculates standard deviation of measurements

        Measurement errors on the measured data; options are relative and
        absolute.

        The second component of the FIM is the measurement error in the form
        of the inverse measurement error covariance matrix Q−1

        It should also be mentioned that correlations between measurements
        can also be specified using this covariance matrix. Throughout this
        package it will be assumed that the error characteristics of the
        measurements can be described in a relatively simple way:
            by asolute or relative errors or by Ternbach
        description.

        Typically, Q is chosen as the inverse of the measurement error
        covariance matrix(Marsili–Libelli et al., 2003; Omlin and Reichert,
        1999; Vanrolleghem and Dochain, 1998)

        Error values are also added to the data_dict version.

        Parameters
        -----------
        meas_error_dict : dict
            dictionary with the variable names and their corresponding errors
            (relative or absolute)
        method : relative|absolute
            relative is percentage value of the variable itself; absolute
            is a constant measurement value

        Notes
        -----
        For the Ternbach method,  the standard deviations of the measurements
        were calculated by:

        .. math:: \sigma_y = \hat{y} \cdot \varsigma_y \cdot
            \left(1+\frac{1}{(\frac{\hat{y}}{lb_y})^2
            + \frac{\hat{y}}{lb_y}} \right)

        Here, :math:`\varsigma_y` and :math:`lb_y` respectively represent a
        constant minimal relative error and a lower accuracy bound on the
        measurement of y. In this way, the standard deviations of the
        meaurements are proportional to the value of the measurements
        :math:`\hat{y}`.
        '''

        if not set(meas_error_dict.keys()).issubset(set(self._variables)):
            raise Exception('Variable {} not listed in current measurements.')

        error_dict = {}

        if method == 'absolute':
            for var in self._variables:
                measerr = meas_error_dict[var]
                error_dict[var] = ERROR_FUN[method].format(str(measerr))

        elif method == 'relative':
            for var in self._variables:
                measerr = meas_error_dict[var]
                error_dict[var] = ERROR_FUN[method].format(str(measerr), var)

        elif method == 'Ternbach':
            for var in self._variables:
                measerr = meas_error_dict[var]
                error_dict[var] = ERROR_FUN[method].format(
                    var, str(lower_accuracy_bound),
                    str(minimal_relative_error))

        elif method == 'direct':
            for var in self._variables:
                measerr = meas_error_dict[var]
                error_dict[var] = measerr

        else:
            raise Exception('Method not implemented!')

        self._uncertainty = Uncertainty(error_dict)
