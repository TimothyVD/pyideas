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

from parameterdistribution import ModPar
from pyideas.modelbase import BaseModel
from pyideas.model import Model
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
            initial_list = self.model._ordered_var['ode']
        except KeyError:
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

    def _dof_dict_to_array_generic(self, dof, dof_dict):
        """
        """
        dof_list = []

        for key in dof:
            if isinstance(dof_dict[key], float) or \
               isinstance(dof_dict[key], int):
                dof_list.append([dof_dict[key]])
            else:
                dof_list.append(*list(dof_dict[key]))

        dof_list = self._flatten_list(dof_list)

        return np.array(dof_list)

    def _dof_dict_to_array(self, dof_dict):
        """
        """
        return self._dof_dict_to_array_generic(self._dof, dof_dict)

    def _dof_array_to_dict_generic(self, dof_len, dof_ordered, dof_array):
        """
        """
        # A FIX FOR CERTAIN SCIPY MINIMIZE ALGORITHMS
        dof_array = dof_array.flatten()

        split_array = np.split(dof_array, np.cumsum(dof_len[:-1]))

        dof_dict = {'parameters': {},
                    'independent': {},
                    'initial': {}}

        if bool(dof_len[0]):
            dof_dict['parameters'].update(dict(zip(dof_ordered['parameters'],
                                                   split_array[0])))
        if bool(dof_len[1]):
            dof_dict['initial'].update(dict(zip(dof_ordered['initial'],
                                                split_array[1])))
        if bool(dof_len[2]):
            # Necessary in case of multiple independent
            indep_split = np.split(split_array[2], dof_len[-1])

            dof_dict['independent'].update(dict(zip(dof_ordered['independent'],
                                                    indep_split)))
        return dof_dict

    def _dof_array_to_dict(self, dof_array):
        """
        """
        return self._dof_array_to_dict_generic(self._dof_len,
                                               self._dof_ordered,
                                               dof_array)

    def _dof_dict_to_model(self, dof_dict):
        """Helper function to pass dof to model
        bool functions are necessary, especially for the independent to avoid
        overwriting of current independent sets by empty dicts
        """
        if bool(dof_dict['parameters']):
            self.model.parameters = dof_dict['parameters']
        if bool(dof_dict['initial']):
            self.model.initial_conditions = dof_dict['initial']
        if bool(dof_dict['independent']):
            self.model.independent = dof_dict['independent']

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

        return self.model._run()

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

    def _basinhopping(self, obj_fun, dof_array, *args, **kwargs):
        '''
        Wrapper for scipy.optimize.basinhopping
        '''
        optimize_info = optimize.basinhopping(obj_fun, dof_array, **kwargs)

        return optimize_info

    def _set_modmeas(self, modeloutput, measurements):
        measurements = pd.DataFrame(measurements,
                                    columns=self.model._variables_of_interest)
        modeloutput = pd.DataFrame(modeloutput,
                                   columns=self.model._variables_of_interest)
        modmeas = pd.concat((measurements, modeloutput), axis=1,
                            keys=['Measured', 'Modelled'])
        index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                          names=self.model._independent_names)
        modmeas.index = index
        self.modmeas = modmeas
#==============================================================================
#         self.modmeas = pd.concat((measurements, modeloutput), axis=1,
#                                  keys=['Measured', 'Modelled'])
#==============================================================================

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
                                    'parameter' % dof.name)
        else:
            raise Exception("Bad input type, give list of ModPar instances.")

    def _set_dof_boundaries(self):
        """
        Define the minimum and maximum boundaries for the different dofs
        """
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
    def _inspyred_bounder(self, candidates, args):
        if self._dof_lower_bnd is None or self._dof_upper_bnd is None:
            raise Exception(('Something went wrong with setting '
                             'self._dof_lower_bnd or self._dof_upper_bnd'))
        candidates = np.array(candidates)

        candidates = np.minimum(np.maximum(candidates, self._dof_lower_bnd),
                                self._dof_upper_bnd)

        return candidates

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

        self._set_dof_boundaries()

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

    def __init__(self, model, measurements, optim_par=None):
        super(self.__class__, self).__init__(model)

        self.measurements = measurements

        if optim_par is None:
            optim_par = self.model.parameters.keys()
        self.dof = optim_par

        self._set_independent()

        self.model.variables_of_interest = self.measurements._variables

        self._minvalues = None
        self._maxvalues = None

    def _set_independent(self):
        """

        """
        # If model is type Model, only 1 independent variable can be selected
        # this means that for ODE models it is forced that the first timestep
        # should occur at timestep 0 instead. This is verified and overruled
        # if necessary
        if isinstance(self.model, Model):
            independent = self.measurements._independent_values
            independent_var = self.model._independent_names[0]
            independent_val = self.measurements._independent_values[independent_var]
            # If ODE is not starting at 0, force it to be so!
            if independent_val[0] != 0.:
                independent[independent_var] = np.insert(independent_val,
                                                         0., 0.)
        elif isinstance(self.model, BaseModel):
            independent = self.measurements._independent_values
        else:
            raise Exception('This model type is not supported!')

        self.model.independent = independent

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

        self._set_modmeas(self._run_model(), self.measurements._data)

        return optimize_info

    def basinhopping(self, pardict=None, obj_crit='wsse', *args, **kwargs):
        '''
        Wrapper for scipy.optimize.minimize
        '''
        def inner_obj_fun(parray=None):
            return self._obj_fun(obj_crit, parray=parray)

        if pardict is None:
            pardict = self.model.parameters.copy()

        optimize_info = \
            self._basinhopping(inner_obj_fun,
                               self._dof_dict_to_array(pardict),
                               *args, **kwargs)

        self._set_modmeas(self._run_model(), self.measurements._data)

        return optimize_info

    def _obj_fun(self, obj_crit, parray=None):
        """
        """
        # Run model
        model_output = self._run_model(dof_array=parray)
        # model_output.sort_index(inplace=True)
        data_output = self.measurements._data
        # data_output.sort_index(inplace=True)

        obj_val = OBJECTIVE_FUNCS[obj_crit](model_output, data_output,
                                            1./self.measurements.meas_uncertainty)

        return obj_val

    def inspyred_optimize(self, obj_crit='wsse', prng=None, approach='PSO',
                          initial_parset=None, add_plot=True, pop_size=16,
                          max_eval=256, **kwargs):
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

        self._set_modmeas(self._run_model(), self.measurements._data)

        return final_pop, ea


class MultiParameterOptimisation(ParameterOptimisation):
    """
    UNDER CONSTRUCTION, ONLY USEFUL FOR LOTS OF DATA OF ODES (WITH DIFFERENT
    INITIAL CONDITIONS)
    """
    def __init__(self, model, measurements, optim_par=None,
                 independent_var='t'):
        super(ParameterOptimisation, self).__init__(model)

        self.measurements = measurements
        self._independent_var = independent_var

        if optim_par is not None:
            self.dof = optim_par
        else:
            self.dof = self.model.parameters.keys()

        #
        measurement_index = measurements._data_index
        drop_independent_level = \
            measurement_index.droplevel(self._independent_var)
        # Keep unique initial conditions
        self.conditions = {}
        self.conditions['values'] = drop_independent_level.unique()
        # Save order of values
        self.conditions['names'] = drop_independent_level.names

    def _run_model(self, dof_array=None):
        '''
        ATTENTION: Zero-point also added, need to be excluded for optimization
        '''
        all_model_output = None
        output_start = 0

        if dof_array is not None:
            # Set new parameters values
            dof_dict = self._dof_array_to_dict(dof_array)
            self._dof_dict_to_model(dof_dict)

        for init_vals in self.conditions['values']:
            init_cond = dict(zip(self.conditions['names'], init_vals))
            self.model.set_initial(init_cond)

            # Get independent values from pandas dataframe
            indep_val = np.array(self.measurements._input_data.xs(init_vals,
                                                                  level=self.conditions['names']).index)
            if indep_val[0] != 0.0:
                output_start = 1
                indep_val = np.concatenate([np.array([0.]), indep_val])
            self.model.set_independent({self.model.independent[0]: indep_val})

            model_output = self.model._run()

            if all_model_output is None:
                all_model_output = model_output.iloc[output_start:]
            else:
                all_model_output = pd.concat([all_model_output,
                                              model_output.iloc[output_start:]],
                                             axis=0)

        return all_model_output.reset_index(drop=True)

#    def _set_independent(self, independent_val):
#        """
#        """
#        independent_dict = {}
#        independent_dict[self._independent_var] = independent_val
#        self.model.set_independent(independent_dict)
