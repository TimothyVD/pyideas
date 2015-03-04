# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:03:10 2015

@author: timothy
"""
import numpy as np
import pandas as pd

from biointense.optimisation import BaseOptimisation
#from sklearn.utils.extmath import cartesian

def A_criterion(FIM):
    '''OED design A criterion
    With this criterion, the trace of the inverse of the FIM is minimized,
    which is equivalent to minimizing the sum of the variances of the
    parameter estimates. In other words, this criterion minimizes the
    arithmetic average of the variances of the parameter estimate.
    Because this criterion is based on an inversion of the FIM,
    numerical problems will arise when the FIM is close to singular.
    '''
    return np.linalg.inv(FIM).trace()

def modA_criterion(FIM):
    '''OED design modified A criterion
    With this criterion, the trace of the inverse of the FIM is minimized,
    which is equivalent to minimizing the sum of the variances of the
    parameter estimates. In other words, this criterion minimizes the
    arithmetic average of the variances of the parameter estimate.
    Because this criterion is based on an inversion of the FIM,
    numerical problems will arise when the FIM is close to singular.
    '''
    return FIM.trace(axis1=-2, axis2=-1)

def D_criterion(FIM):
    '''OED design D criterion
    Here, the idea is to maximize the determinant of the FIM
    (Box and Lucas, 1959). The latter is inversely proportional to the
    volume of the confidence region of the parameter es- timates, and this
    volume is thus minimized when maximizing det (FIM). In other words,
    one minimizes the geometric average of the variances of the parameter
    estimates. More- over, D-optimal experiments possess the property of
    being invariant with respect to any rescaling of the parameters
    (Petersen, 2000; Seber and Wild, 1989). According to Walter and
    Pronzato (1997), the D-optimal design criterion is the most used
    criterion. However, several authors have pointed out that this
    criterion tends to give excessive importance to the parameter which
    is most influential.
    '''
    return np.linalg.det(FIM)

def E_criterion(FIM):
    '''OED design E criterion
    The E-optimal design criterion maximizes the smallest eigenvalue of
    the FIM and thereby minimizes the length of the largest axis of the
    confidence ellipsoid. Thus, these designs aim at minimizing the
    largest parameter estimation variance and thereby at maximizing the
    distance from the singular, unidentifiable case.
    '''
    return np.min(np.linalg.eigvals(FIM), axis=-1)

def modE_criterion(FIM):
    '''OED design modE criterion
    With this criterion, the focus is on the minimization of the condition
    number, which is the ratio between the largest and the smallest
    eigenvalue, or, in other words, the ratio of the shortest and the
    longest ellipsoid axes. The minimum of this ratio is one, which corresponds
    to the case where the shape of the confidence ellipsoid
    is a (hyper)sphere.
    '''
    w = np.linalg.eigvals(FIM)
    return np.max(w, axis=-1)/np.min(w, axis=-1)

OED_CRITERIA = {'A': A_criterion, 'modA': modA_criterion, 'D': D_criterion,
                'E': E_criterion, 'modE': modE_criterion}

OED_CRITERIA_MAXIMIZE = {'A': False, 'modA': True, 'D': True,
                         'E': True, 'modE': False}

class BaseOED(BaseOptimisation):
    """
    """

    def __init__(self, confidence):
        super(BaseOED, self).__init__()
        self.confidence = confidence
        self.model = confidence.model

#        self._optim_par = []
#        Dof distributions
#        # ODE
#        self._dof = {'initial': {'SA': dist},
#                     'independent': {'t': dist}}
#        # Algebraic
#        self._dof = {'initial': {},
#                     'independent': {'SA': dist,
#                                     'SB': dist}}
        self._dof = {'initial': {},
                     'independent': {},
                     'parameter': {}}

        self._initial = []
        self._len_initial = 0
        self._independent = []
        self._len_independent = 0
        self._parameter = []
        self._len_parameter = 0

        self._independent_samples = 0

        self._criterion = 'D'

    def set_degrees_of_freedom(self, dof_dict, samples):
        """
        """
        initial = dof_dict.get('initial', {})
        self._len_initial = len(initial)
        self._initial = initial.keys()
        independent = dof_dict.get('independent', {})
        self._len_independent = len(independent)
        self._independent = independent.keys()
        parameter = dof_dict.get('parameter', {})
        self._len_parameter = len(parameter)
        self._parameter = parameter.keys()
        if self._len_initial > 0 and self._len_independent > 0:
            raise Exception('A multidimensional model cannot have initial '
                            'values!')

        self._dof = {'initial': {},
                     'independent': {},
                     'parameter': {}}

        self._independent_samples = samples

        if isinstance(dof_dict, dict):
            if bool(initial):
                self._dof['initial'] = initial
            if bool(independent):
                self._dof['independent'] = independent
            if bool(parameter):
                self._dof['parameter'] = parameter

        else:
            raise Exception('optim_par needs to be a dict!')

    def _array_to_initial(self, array):
        """
        """
        for i, key in enumerate(self._initial):
            self.model.set_initial({key: array[i]})

    def _array_to_independent(self, array):
        """
        """
        array = np.reshape(array, [self._len_independent,
                                   self._independent_samples])

        for i, key in enumerate(self._independent):
            self.model.set_independent({key: array[i, :]})

    def _array_to_parameter(self, array):
        """
        """
        for i, key in enumerate(self._parameter):
            self.model.set_parameters({key: array[i]})

    def _array_to_model(self, array):
        """
        """
        # A FIX FOR CERTAIN SCIPY MINIMIZE ALGORITHMS
        array = array.flatten()

        self._array_to_initial(array[:self._len_initial])
        self._array_to_parameter(array[self._len_initial:self._len_initial + self._len_parameter])
        self._array_to_independent(array[self._len_initial + self._len_parameter:])

    def _bounder(self):
        '''
        Genere
        '''
        minsample = []
        maxsample = []
        #use get_fitting_parameters, since this is ordered dict!!
        for initial in self._initial:
            minsample.append([self._dof['initial'][initial].min])
            maxsample.append([self._dof['initial'][initial].max])
        for parameter in self._parameter:
            minsample.append([self._dof['parameter'][parameter].min])
            maxsample.append([self._dof['parameter'][parameter].max])
        for independent in self._independent:
            minsample.append([self._dof['independent'][independent].min] *
                              self._independent_samples)
            maxsample.append([self._dof['independent'][independent].max] *
                              self._independent_samples)
        minsample = np.array([y for x in minsample for y in x])
        maxsample = np.array([y for x in maxsample for y in x])
        return minsample, maxsample

    def _bounder_generator(self, candidates, args):
        candidates = np.array(candidates)

        candidates = np.minimum(np.maximum(candidates, self._minvalues),
                                self._maxvalues)

        return candidates

    def _sample_generator(self, random, args):
        '''
        '''
        samples = []
        #use get_fitting_parameters, since this is ordered dict!!
        for initial in self._initial:
            samples.append([self._dof['initial'][initial].aValue()])
        for parameter in self._parameter:
            samples.append([self._dof['parameter'][parameter].aValue()])
        for independent in self._independent:
            samples.append(list(self._dof['independent'][independent].MCSample(
                self._independent_samples)))
        samples = [y for x in samples for y in x]
        return samples

    def _run_confidence(self, array=None):
        '''
        ATTENTION: Zero-point also added, need to be excluded for optimization
        '''
        #run option
        if array is not None:
            #run model first with new parameters
            pardict = self._array_to_model(array)

    def _obj_fun(self, obj_fun, array=None):
        """
        """
        # Run model
        modeloutput = self._run_confidence(array=array)

        obj_val = OED_CRITERIA[obj_fun](self.confidence.FIM)

        return obj_val

    def _obj_fun_inspyred(self, obj_fun, candidates, args):
        '''
        '''
        fitness = []
        for cs in candidates:
            fitness.append(self._obj_fun(obj_fun, array=np.array(cs)))
        return fitness

    def bioinspyred_optimize(self, criterion='D', prng=None, approach='PSO',
                             initial_parset=None, pop_size=16, max_eval=256,
                             **kwargs):
        """
        """
        self._criterion = criterion

        self._minvalues, self._maxvalues = self._bounder()

        final_pop, ea = self._bioinspyred_optimize(obj_fun=criterion,
                                                   prng=prng,
                                                   approach=approach,
                                                   initial_parset=initial_parset,
                                                   pop_size=pop_size,
                                                   maximize=OED_CRITERIA_MAXIMIZE[criterion],
                                                   max_eval=max_eval, **kwargs)

        return final_pop, ea

    def select_optimal_individual(self, final_pop):
        '''
        '''
        if type(final_pop)!= list:
            raise Exception('final_pop has to be a list!')

        if OED_CRITERIA_MAXIMIZE[self._criterion]:
            print('Individual with maximum fitness is selected!')
            return max(final_pop)
        else:
            print('Individual with minimum fitness is selected!')
            return min(final_pop)

    def brute_oed(self, step_dict, criterion='D'):
        """
        """
        self._criterion = criterion

        independent_dict = {}
        for independent in step_dict.keys():
            independent_dict[independent] = \
                np.linspace(self._dof['independent'][independent].min,
                            self._dof['independent'][independent].max,
                            step_dict[independent])

        self.model.set_independent(independent_dict)

        index = pd.MultiIndex.from_tuples(zip(*self.model._independent_values.values()),
                                          names=self.model.independent)

        if OED_CRITERIA_MAXIMIZE[criterion]:
            selection_criterion = np.argmax
        else:
            selection_criterion = np.argmin

        FIM_evolution = self.confidence.FIM_time

        FIM_tot = 0
        experiments = []
        for i in range(self._independent_samples):
            OED_criterion = OED_CRITERIA[criterion](FIM_evolution)

            optim_indep = selection_criterion(OED_criterion)
            experiments.append(index[optim_indep])

            FIM_tot += FIM_evolution[optim_indep, :, :]
            FIM_evolution = FIM_evolution + FIM_evolution[optim_indep, :, :]

        return np.array(experiments), FIM_tot
