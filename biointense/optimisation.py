# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:19:34 2015

@author: timothy
"""

from scipy import optimize
import numpy as np


class BaseOptimisation(object):
    """
    """

    def __init__(self):
        self._obj_fun = None

        self._optim_par = None


    @property
    def optim_par(self):
        """
        """
        return self._optim_par

    @optim_par.setter
    def optim_par(self, par_to_optim):
        """
        """
        return NotImplementedError

    def _local_optimize(self, obj_func, parray, method, **kwargs):
        '''
        Wrapper for scipy.optimize.minimize
        '''

        optimize_info = optimize.minimize(_obj_fun, parray,
                                          method=method, *args, **kwargs)

        return optimize_info

    def _obj_fun(self, parray):
        '''
        '''
        return NotImplementedError


class ParameterOptimisation(BaseOptimisation):
    """
    """

    def __init__(self, model, measurements, par_to_opt):
        super(ParameterOptimisation).__init__()

        self.model = model
        self.measurements = measurements

        self.optim_par = par_to_opt

    @property
    def optim_par(self):
        """
        """
        return self._par_to_opt

    @optim_par.setter
    def optim_par(self, par_to_optim):
        """
        """
        if isinstance(par_to_optim, dict):
            self._par_to_opt = par_to_opt.keys()
            for key in self.par_to_opt:
                self.model.parameters[key] = par_to_optim[key]
        elif isinstance(par_to_optim, list):
            self._par_to_opt = par_to_opt
        else:
            raise Exception('par_to_opt needs to be dict or list!')

    def _pardict_to_pararray(self, pardict):
        """
        """
        pararray = np.zeros(len(pardict))
        for i, key in enumerate(self.optim_par):
            pararray[i] = pardict[key]

        return pararray

    def _pararray_to_pardict(self, pararray):
        """
        """
        # A FIX FOR CERTAIN SCIPY MINIMIZE ALGORITHMS
        pararray = pararray.flatten()

        for i, key in enumerate(self.optim_par):
            pardict[key] = pararray[i]

        return pardict

    def local_optimize(self, pardict=None, method='Nelder-Mead',
                       *args, **kwargs):
        '''
        Wrapper for scipy.optimize.minimize
        '''

        optimize_info = \
            self._local_optimize(self.get_WSSE,
                                pararray=self._pardict_to_pararray(pardict),
                                method=method, *args, **kwargs)

        return optimize_info

    @staticmethod
    def _sse(*args):
        """
        """
        #residuals =
        modeloutput = args[0]
        data = args[1]

        residuals = (modeloutput - data)

        sse = (residuals * residuals).sum().sum()

        return sse

    def get_obj_fun(self, obj_fun, pararray=None):
        """
        """
        # Run model
        modeloutput = self._run_model(pararray=pararray)

        obj_val = self.__getattribute__('_' + obj_fun)(modeloutput,
                                                       self.measurements.data,
                                                       self.measurements.weights)

        return obj_val

    def _run_model(self, pararray=None):
        '''
        ATTENTION: Zero-point also added, need to be excluded for optimization
        '''
        #run option
        if pararray is not None:
            #run model first with new parameters
            pardict = self._pararray_to_pardict(pararray)
            self.model.set_parameters(pardict)

        return self.model.run()


class OEDOptimisation(BaseOptimisation):
    """
    """

    def __init__(self, oed):
        super(OEDOptimisation).__init__()

