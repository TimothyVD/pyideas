# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:19:34 2015

@author: timothy
"""

from scipy import optimize
import numpy as np
import pandas as pd
try:
    import inspyred  # Global optimization
    INSPYRED_IMPORT = True
    INSPYRED_APPROACH = {'PSO': inspyred.swarm.PSO, 'DEA': inspyred.ec.DEA,
                         'SA': inspyred.ec.SA}
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


class _BaseOptimisation(object):
    """
    """

    def __init__(self, model):
        self.model = model
        self._dof_model = self._get_dof_model()
        self._dof = None
        self._dof_ordered = None
        # [Parameters, Initial, ]
        self._how_to_order_dof = ['parameters', 'initial', 'independent']
        self._dof_len = [0, 0, 0]

        self._distributions_set = False
        self._dof_distributions = None
        self.modmeas = None
        self._independent_samples = None

    @staticmethod
    def _flatten_list(some_list):
        return [item for sublist in some_list for item in sublist]

    def _get_dof_model(self):
        """
        """
        try:
            initial_list = self.model.initial
        except AttributeError:
            initial_list = []

        _dof_list = self._flatten_list([self.model.parameters.keys(),
                                        self.model.independent,
                                        initial_list])

        _dof_ref = {}.fromkeys(_dof_list, None)
        _dof_ref.update({}.fromkeys(self.model.parameters.keys(),
                                    'parameters'))

        _dof_ref.update({}.fromkeys(self.model.independent,
                                    'independent'))
        _dof_ref.update({}.fromkeys(initial_list,
                                    'initial'))

        return _dof_ref

    @property
    def dof(self):
        """
        """
        return self._dof

    @dof.setter
    def dof(self, dof_list):
        """
        """
        self._dof = []
        self._dof_len = []
        self._dof_ordered = {'parameters': [],
                             'independent': [],
                             'initial': []}

        # Select dof according to subgroup they belong to
        for dof in dof_list:
            self._dof_ordered[self._dof_model[dof]].append(dof)

        # Append dof in certain order
        for subgroup in self._how_to_order_dof:
            self._dof.append(self._dof_ordered[subgroup])
            # Calc length for each subgroup
            self._dof_len.append(len(self._dof_ordered[subgroup]))

        self._dof = self._flatten_list(self._dof)

        self._dof_lower_bnd = None
        self._dof_upper_bnd = None

    def _dof_dict_to_array(self, dof_dict):
        """
        """
        dof_list = []

        for key in self.dof:
            if isinstance(dof_dict[key], float):
                dof_list.append([dof_dict[key]])
            else:
                dof_list.append(*list(dof_dict[key]))

        dof_list = self._flatten_list(dof_list)

        return np.array(dof_list)

    def _dof_array_to_dict(self, dof_array):
        """
        """
        # A FIX FOR CERTAIN SCIPY MINIMIZE ALGORITHMS
        dof_array = dof_array.flatten()

        split_array = np.split(dof_array, np.cumsum(self._dof_len[:-1]))

        dof_dict = {'parameters': {},
                    'independent': {},
                    'initial': {}}

        if bool(self._dof_len[0]):
            dof_dict['parameters'].update(dict(zip(self._dof_ordered['parameters'],
                                                   split_array[0])))
        if bool(self._dof_len[1]):
            dof_dict['initial'].update(dict(zip(self._dof_ordered['initial'],
                                                split_array[1])))
        if bool(self._dof_len[2]):
            # Necessary in case of multiple independent
            indep_split = np.split(split_array[2], self._dof_len[-1])

            dof_dict['independent'].update(dict(zip(self._dof_ordered['independent'],
                                                    indep_split)))
        return dof_dict

    def _dof_dict_to_model(self, dof_dict):
        """
        """
        if bool(self._dof_len[0]):
            self.model.set_parameters(dof_dict['parameters'])
        if bool(self._dof_len[1]):
            self.model.set_initial(dof_dict['initial'])
        if bool(self._dof_len[2]):
            self.model.set_independent(dof_dict['independent'])

    def _dof_array_to_model(self, dof_array):
        """
        """
        dof_dict = self._dof_array_to_dict(dof_array)
        self._dof_dict_to_model(dof_dict)


    def _run_model(self, dof_array=None):
        '''
        ATTENTION: Zero-point also added, need to be excluded for optimization
        '''
        if dof_array is not None:
            # Set new parameters values
            dof_dict = self._dof_array_to_dict(dof_array)
            self._dof_dict_to_model(dof_dict)

        return self.model.run()

    def _obj_fun(self, obj_crit, parray):
        '''
        '''
        # Run model

        # Evaluate model

        return NotImplementedError

    def _local_optimize(self, obj_fun, dof_array, method, *args, **kwargs):
        '''
        Wrapper for scipy.optimize.minimize
        '''
        optimize_info = optimize.minimize(obj_fun, dof_array,
                                          method=method, **kwargs)

        return optimize_info

    def _set_modmeas(self, modeloutput, measurements):
        self.modmeas = pd.concat((measurements, modeloutput), axis=1,
                                 keys=['Measured', 'Modelled'])

    def set_dof_distributions(self, dof_dist_list):
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
        # Checking if distirbutions are already set
        if not self._distributions_set:
            self._dof_distributions = {}
            self._distributions_set = True

        if isinstance(dof_dist_list, ModPar):  # one parameter
            dof_dist_list = [dof_dist_list]

        if isinstance(dof_dist_list, list):
            # A list of ModPar instances
            for dof in dof_dist_list:
                if dof.name in self.dof:
                    self._dof_distributions[dof.name] = dof
                else:
                    raise Exception('Parameter %s is not listed as fitting '
                                    'parameter' % parameter.name)
        else:
            raise Exception("Bad input type, give list of ModPar instances.")

        self._set_dof_boundaries()

    def _set_dof_boundaries(self):
        '''
        '''
        minsample = []
        maxsample = []
        for dof in self.dof:
            dof_min = [self._dof_distributions[dof].min]
            dof_max = [self._dof_distributions[dof].max]
            if self._dof_model[dof] is 'independent':
                dof_min *= self._independent_samples
                dof_max *= self._independent_samples
            minsample.append(dof_min)
            maxsample.append(dof_max)
        self._dof_lower_bnd = np.array(self._flatten_list(minsample))
        self._dof_upper_bnd = np.array(self._flatten_list(maxsample))


    # Bioinspyred specific stuff

    def _inspyred_bounder(self):
        '''
        '''
        return NotImplementedError

    def _inspyred_sampler(self, random, args):
        '''
        '''
        samples = []
        # use get_fitting_parameters, since this is ordered dict!!
        for dof in self.dof:
            if self._dof_model[dof] is not 'independent':
                samples.append([self._dof_distributions[dof].aValue()])
            else:
                samples.append(list(self._dof_distributions[dof].MCSample(
                    self._independent_samples)))
        return self._flatten_list(samples)

    def _inspyred_obj_fun(self, obj_fun, candidates, args):
        '''
        '''
        fitness = []
        for cs in candidates:
            fitness.append(obj_fun(parray=np.array(cs)))
        return fitness

    def _inspyred_optimize(self, obj_fun, **kwargs):
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

        if kwargs.get('approach') in INSPYRED_APPROACH:
            ea = INSPYRED_APPROACH[kwargs.get('approach')](prng)
        else:
            raise Exception('This approach is currently not supported!')

        if kwargs.get('approach') == 'PSO':
            ea.topology = inspyred.swarm.topologies.ring_topology

        def temp_get_objective(candidates, args):
            return self._inspyred_obj_fun(obj_fun, candidates, args)

        ea.terminator = inspyred.ec.terminators.evaluation_termination
        final_pop = ea.evolve(generator=self._inspyred_sampler,
                              evaluator=temp_get_objective,
                              pop_size=kwargs.get('pop_size'),
                              bounder=self._inspyred_bounder,
                              maximize=kwargs.get('maximize'),
                              max_evaluations=kwargs.get('max_eval'),
                              neighborhood_size=5)

        final_pop.sort(reverse=True)
        return final_pop, ea


class ParameterOptimisation(_BaseOptimisation):
    """
    """

    def __init__(self, model, measurements, optim_par=None,
                 overwrite_independent=False):
        super(ParameterOptimisation, self).__init__(model)

        self.measurements = measurements

        if optim_par is not None:
            self.dof = optim_par
        else:
            self.dof = self.model.parameters.keys()

        self._set_independent(overwrite_independent)

        self._minvalues = None
        self._maxvalues = None

    def _inspyred_bounder(self, candidates, args):
        candidates = np.array(candidates)

        candidates = np.minimum(np.maximum(candidates, self._dof_lower_bnd),
                                self._dof_upper_bnd)

        return candidates

    def _set_independent(self, overwrite_independent):
        """
        """
        independent = self.measurements.Data.index.names
        independent_val = self.measurements.Data.index

        independent_dict = {}
        for key in independent:
            if overwrite_independent:
                independent_dict[key] = \
                    independent_val.get_level_values(key).values
            else:
                independent_dict[key] = \
                    np.append(self.model._independent_values[key],
                              independent_val.get_level_values(key).values)
        # Check for duplicates and delete those
        independent_pd = pd.DataFrame(independent_dict)
        # Remove any duplicates
        independent_pd = independent_pd.drop_duplicates()
        # Make sure the values are ordered (important for ODEs)
        independent_pd = independent_pd.sort(columns=independent)
        self.model.set_independent(independent_pd)

    def local_optimize(self, pardict=None, obj_crit='wsse',
                       method='Nelder-Mead', *args, **kwargs):
        '''
        Wrapper for scipy.optimize.minimize
        '''
        def inner_obj_fun(parray=None):
            return self._obj_fun(obj_crit, parray=parray)

        if pardict is None:
            pardict = self.model.parameters.copy()

        optimize_info = \
            self._local_optimize(inner_obj_fun,
                                 self._dof_dict_to_array(pardict),
                                 method, *args, **kwargs)

        self._set_modmeas(self.model.run(), self.measurements.Data)

        return optimize_info

    def _obj_fun(self, obj_crit, parray=None):
        """
        """
        # Run model
        modeloutput = self._run_model(dof_array=parray)

        obj_val = OBJECTIVE_FUNCS[obj_crit](modeloutput,
                                            self.measurements.Data,
                                            1./self.measurements._Error_Covariance_Matrix_PD)

        return obj_val

    def inspyred_optimize(self, obj_crit='wsse', prng=None, approach='PSO',
                             initial_parset=None, add_plot=True,
                             pop_size=16, max_eval=256, **kwargs):
        """
        """
        def inner_obj_fun(parray=None):
            return self._obj_fun(obj_crit, parray=parray)

        final_pop, ea = self._inspyred_optimize(inner_obj_fun,
                                                prng=prng,
                                                approach=approach,
                                                initial_parset=initial_parset,
                                                add_plot=add_plot,
                                                pop_size=pop_size,
                                                maximize=False,
                                                max_eval=max_eval, **kwargs)

        self._set_modmeas(self.model.run(), self.measurements.Data)

        return final_pop, ea
