# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:19:34 2015

@author: timothy
"""

from scipy import optimize
import numpy as np
try:
    import inspyred  # Global optimization
    INSPYRED_IMPORT = True
except:
    INSPYRED_IMPORT = False

from parameterdistribution import *
from time import time
from random import Random


def wsse(*args):
    """
    """
    modeloutput = args[0]
    data = args[1]
    weights = args[2]

    residuals = (modeloutput - data)

    wsse = (residuals * weights * residuals).sum().sum()

    return wsse

def sse(*args):
    """
    """
    sse = wsse(args[0], args[1], 1.0)

    return sse


OBJECTIVE_FUNCS = {'wsse': wsse, 'sse': sse}


class BaseOptimisation(object):
    """
    """

    def __init__(self):
        self._optim_par = None
        self._distributions_set = False


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

    def _local_optimize(self, obj_fun, parray, method, *args, **kwargs):
        '''
        Wrapper for scipy.optimize.minimize
        '''

        optimize_info = optimize.minimize(obj_fun, parray,
                                          method=method, *args, **kwargs)

        return optimize_info

    def _obj_fun(self, parray):
        '''
        '''
        return NotImplementedError

    def _sample_generator(self, random, args):
        '''
        '''
        samples = []
        #use get_fitting_parameters, since this is ordered dict!!
        for parameter in self.optim_par:
            samples.append(self.pardistributions[parameter].aValue())
        return samples

    def set_fitting_par_distributions(self, pardistrlist):
        """
        For each parameter set as fitting parameter, the information
        of the distribution is set.

        Parameters
        ------------
        pardistrlist : list
            List of ModPar instances
        optguess : boolean
            Put this True if you want to update the currently saved optimal
            guess value for the parameters

        """
        #Checking if distirbutions are already set
        if not self._distributions_set:
            self.pardistributions = {}
            self._distributions_set = True

        if isinstance(pardistrlist,ModPar): #one parameter
            if len(self.optim_par) > 1:
                raise Exception("""Only one parameterdistirbution is given,
                whereas the number of fitting parameters is %d
                """ %len(self.optim_par))
            else:
                if pardistrlist.name in self.optim_par:
                    par_value = self.model.__getattribute__(self._optim_ref[parameter.name])[parameter.name]
                    if not pardistrlist.min < par_value < pardistrlist.max:
                        raise Exception('Current parvalue is not between min and max value of the parameter!')
                    if pardistrlist.name in self.pardistributions:
                        if self._print_on:
                            print('Parameter distribution info updated for %s' %pardistrlist.name)
                        self.pardistributions[pardistrlist.name] = pardistrlist
                    else:
                        self.pardistributions[pardistrlist.name] = pardistrlist
                else:
                    raise Exception('Parameter is not listed as fitting parameter')

        elif isinstance(pardistrlist,list):
            #A list of ModPar instances
            for parameter in pardistrlist:
                if parameter.name in self.optim_par:
                    par_value = self.model.__getattribute__(self._optim_ref[parameter.name])[parameter.name]
                    if not parameter.min < par_value < parameter.max:
                        raise Exception('Current parvalue is not between min and max value of the parameter!')
                    if parameter.name in self.pardistributions:
                        print('Parameter distribution info updated for %s' %parameter.name)
                        self.pardistributions[parameter.name] = parameter
                    else:
                        self.pardistributions[parameter.name] = parameter
                else:
                    raise Exception('Parameter %s is not listed as fitting parameter' %parameter.name)
        else:
            raise Exception("Bad input type, give list of ModPar instances.")

    def _bounder_generator(self):
        '''
        Genere
        '''
        minsample = []
        maxsample = []
        #use get_fitting_parameters, since this is ordered dict!!
        for parameter in self.optim_par:
            minsample.append(self.pardistributions[parameter].min)
            maxsample.append(self.pardistributions[parameter].max)
        return minsample, maxsample

    def _obj_fun_inspyred(self, obj_fun, candidates, args):
        '''
        '''
        return NotImplementedError

    def _bioinspyred_optimize(self, obj_fun, **kwargs):
        """

        Notes
        ------
        A working version of Bio_inspyred is needed to get this optimization
        running!
        """
        if not INSPYRED_IMPORT:
            raise Exception("Inspyred was not found, no global optimization "
                            "possible!")

        # OPTIMIZATION
        prng = kwargs.get('prng')
        if prng is None:
            prng = Random()
            prng.seed(time())

        if kwargs.get('approach') == 'PSO':
            ea = inspyred.swarm.PSO(prng)
            ea.topology = inspyred.swarm.topologies.ring_topology
        elif kwargs.get('approach') == 'DEA':
            ea = inspyred.ec.DEA(prng)
        elif kwargs.get('approach') == 'SA':
            ea = inspyred.ec.SA(prng)
        else:
            raise Exception('This approach is currently not supported!')

        def temp_get_objective(candidates, args):
            return self._obj_fun_inspyred(obj_fun, candidates, args)

        ea.terminator = inspyred.ec.terminators.evaluation_termination
        final_pop = ea.evolve(generator=self._sample_generator,
                              evaluator=temp_get_objective,
                              pop_size=kwargs.get('pop_size'),
                              bounder=self._bounder_generator,
                              maximize=kwargs.get('maximize'),
                              max_evaluations=kwargs.get('max_eval'),
                              neighborhood_size=5)

        final_pop.sort(reverse=True)
        return final_pop, ea


class ParameterOptimisation(BaseOptimisation):
    """
    """

    def __init__(self, model, measurements, optim_par):
        super(ParameterOptimisation, self).__init__()

        self.model = model
        self.measurements = measurements

        self._optim_par = optim_par
        self._set_optim_ref()

    def _set_optim_ref(self):
        self._optim_ref = {}
        for par in self.optim_par:
            self._optim_ref[par] = 'parameters'

    @property
    def optim_par(self):
        """
        """
        return self._optim_par

    @optim_par.setter
    def optim_par(self, optim_par):
        """
        """
        if isinstance(par_to_opt, dict):
            self._optim_par = optim_par.keys()
            for key in self.optim_par:
                self.model.parameters[key] = optim_par[key]
        elif isinstance(optim_par, list):
            self._optim_par = optim_par
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

        pardict = self.model.parameters.copy()
        for i, key in enumerate(self.optim_par):
            pardict[key] = pararray[i]

        return pardict

    def local_optimize(self, pardict=None, obj_fun='wsse',
                       method='Nelder-Mead', *args, **kwargs):
        '''
        Wrapper for scipy.optimize.minimize
        '''
        if pardict is None:
            pardict = self.model.parameters.copy()

        def temp_obj_fun(pararray=None):
            return self._obj_fun(obj_fun, pararray=pararray)

        optimize_info = \
            self._local_optimize(temp_obj_fun,
                                 self._pardict_to_pararray(pardict),
                                 method, *args, **kwargs)

        return optimize_info

    def _obj_fun(self, obj_fun, pararray=None):
        """
        """
        # Run model
        modeloutput = self._run_model(pararray=pararray)

        obj_val = OBJECTIVE_FUNCS[obj_fun](modeloutput,
                                           self.measurements.Data,
                                           1/self.measurements._Error_Covariance_Matrix_PD)

        return obj_val

    def _obj_fun_inspyred(self, obj_fun, candidates, args):
        '''
        '''
        fitness = []
        for cs in candidates:
            fitness.append(self._obj_fun(obj_fun, pararray=np.array(cs)))
        return fitness

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

    def bioinspyred_optimize(self, obj_fun='wsse', prng=None, approach='PSO',
                             initial_parset=None, add_plot=True,
                             pop_size=16, max_eval=256, **kwargs):
        """
        """

        final_pop, ea = self._bioinspyred_optimize(obj_fun=obj_fun, prng=prng,
                                                   approach=approach,
                                                   initial_parset=initial_parset,
                                                   add_plot=add_plot,
                                                   pop_size=pop_size,
                                                   maximize=False,
                                                   max_eval=max_eval, **kwargs)

        return final_pop, ea




