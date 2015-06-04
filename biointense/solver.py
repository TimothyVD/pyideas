# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 11:52:12 2015

@author: timothy
"""
from __future__ import division

from scipy.integrate import odeint, ode
from itertools import product

# Check whether odespy is installed
try:
    _odespy = True
    import odespy
except:
    _odespy = False
    pass

import pandas as pd

# Define availabe integrators for each of all ode' approaches
ODE_INTEGRATORS = {}
ODE_INTEGRATORS['ode'] = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
if _odespy:
    ODE_INTEGRATORS['odespy'] = odespy.list_available_solvers()

# Standard setting for each ode approach
STD_ODE_INTEGRATOR = {'ode': 'lsoda', 'odespy': 'lsoda_scipy', 'odeint': ''}

def _flatten_list(some_list):
    return [item for sublist in some_list for item in sublist]

class _Solver(object):

    def __init__(self, model):
        """
        """
        self.model = model

    def _check_solver_sanity(self):
        """
        check whether solver setting is compatible with system
        check the external event: can the solver cope with the time step of
        the measurement?
        """

    def solve(self):
        """
        Solve model equations

        return: For each independent value, returns state for all variables
        """
        return NotImplementedError


class OdeSolver(_Solver):
    """
    """
    def __init__(self, model, ode_solver_options=None, ode_integrator=None):
        """
        """
        self.model = model
        self.independent = model._ode_independent
        self.ode_solver_options = ode_solver_options or {}
        self.ode_integrator = ode_integrator
        self._initial_conditions = None
        self._ode_procedure = {'odeint': self._solve_odeint,
                               'ode': self._solve_ode,
                               'odespy': self._solve_odespy}

    def _check_ode_integrator_setting(self, procedure):
        """
        """
        if self.ode_integrator is None:
            self.ode_integrator = STD_ODE_INTEGRATOR[procedure]
        else:
            if procedure is not 'odeint' and \
               self.ode_integrator not in ODE_INTEGRATORS[procedure]:
                raise Exception(self.ode_integrator + ' is not available, '
                                'please choose one from the ODE_INTEGRATORS '
                                'list.')

    def _args_ode_function(self, **kwargs):
        """
        """
        externalfunctions = kwargs.get('externalfunctions')
        if externalfunctions:
            args = (self.model.parameters, externalfunctions,)
        else:
            args = (self.model.parameters,)

        return args

    def _solve_odeint(self, fun_ode, **kwargs):
        """
        Calculate the ode equations using scipy integrate odeint solvers

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from the ode equations in function of the
        independent values

        Notes
        ------
        The scipy integrate odeint module can be found on [1]_.

        References
        -----------
        .. [1] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
        scipy.integrate.odeint.html
        """
        args = self._args_ode_function(**kwargs)

        res = odeint(fun_ode,
                     self._initial_conditions,
                     self.model._independent_values[self.independent],
                     args=args, **self.ode_solver_options)

        return res, self.model._independent_values[self.independent]

    def _solve_ode(self, fun_ode, **kwargs):
        """
        Calculate the ode equations using scipy integrate ode solvers

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from the ode equations in function of the
        independent values

        Notes
        ------
        The scipy integrate ode module can be found on [1]_.

        References
        -----------
        .. [1] http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
        scipy.integrate.ode.html
        """
        self._check_ode_integrator_setting("ode")

        args = self._args_ode_function(**kwargs)

        # Make wrapper function to
        def wrapper(independent_values, initial_conditions, parameters):
            return fun_ode(initial_conditions, independent_values, parameters)

        solver = ode(wrapper).set_integrator(self.ode_integrator,
                                             **self.ode_solver_options)

        solver.set_initial_value(self._initial_conditions,
                                 self.model._independent_values[self.independent][0])
        solver.set_f_params(*args)

        xdata = self.model._independent_values[self.independent]
        timesteps = xdata[1:] - xdata[:-1]
        model_output = []
        xdata = []
        model_output.append(solver.y)
        xdata.append(solver.t)
        for dt in timesteps:
            if solver.successful():
                solver.integrate(solver.t + dt)
                model_output.append(solver.y)
                xdata.append(solver.t)

        return model_output, xdata

    def _solve_odespy(self, fun_ode, **kwargs):
        """
        Calculate the ode equations using scipy integrate ode solvers

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from the ode equations in function of the
        independent values

        Notes
        ------
        The odespy package can be found on [1]_.

        References
        -----------
        .. [1] H. P. Langtangen and L. Wang. Odespy software package.
        URL: https://github.com/hplgit/odespy. 2014
        """
        if not _odespy:
            raise Exception('Odespy is not installed!')

        self._check_ode_integrator_setting("ode")

        args = self._args_ode_function(**kwargs)

        solver = odespy.__getattribute__(self.ode_integrator)
        solver = solver(fun_ode)
        if self.ode_solver_options is not None:
            solver.set(**self.ode_solver_options)
        solver.set_initial_condition(self._initial_conditions)
        solver.set(f_args=args)

        xdata = self.model._independent_values[self.independent]
        model_output, xdata = solver.solve(xdata)

        return model_output, xdata

    def solve(self, procedure='odeint', **kwargs):
        """
        Calculate the ode equations using scipy integrate odeint solvers

        Returns
        -------
        result : pd.DataFrame
            Contains all outputs from the ode equations in function of the
            independent values
        """
        self._initial_conditions = [self.model.initial_conditions[var]
                                    for var in self.model._ordered_var['ode']]
        self._check_ode_integrator_setting(procedure)

        output, xdata = self._ode_procedure[procedure](self.model.fun_ode,
                                                       **kwargs)

        result = pd.DataFrame(output, index=xdata,
                              columns=self.model._ordered_var['ode'])

        return result

    def _solve_direct_lsa(self, fun_ode_lsa, dxdtheta_start,
                          procedure='odeint', **kwargs):
        """
        Calculate the ode equations using scipy integrate odeint solvers

        Returns
        -------
        result : pd.DataFrame
            Contains all outputs from the ode equations in function of the
            independent values
        """
        self._initial_conditions = [self.model.initial_conditions[var]
                                    for var in self.model._ordered_var['ode']]
        self._initial_conditions += _flatten_list(dxdtheta_start.tolist())

        self._check_ode_integrator_setting(procedure)

        output, xdata = self._ode_procedure[procedure](fun_ode_lsa, **kwargs)

        #result = pd.DataFrame(output, index=xdata,
        #                      columns=self.model._ordered_var['ode'])

        return output


class AlgebraicSolver(_Solver):
    """
    Class to calculate the algebraic equations/models
    """
    def _solve_algebraic_generic(self, alg_function, *args, **kwargs):
        """
        """
        model_output = alg_function(self.model._independent_values,
                                    self.model.parameters,
                                    *args, **kwargs)

        return model_output

    def _solve_algebraic(self, *args, **kwargs):
        """
        """

        model_output = self._solve_algebraic_generic(self.model.fun_alg, *args,
                                                     **kwargs)

        # TODO! Is dict for (Algebraic)model, but not for optim
        if isinstance(self.model._independent_values, dict):
            independent = self.model._independent_values.values()
        elif isinstance(self.model._independent_values, pd.DataFrame):
            independent = self.model._independent_values.values
        else:
            raise Exception('Independent needs to be dict of array(s)!')
        index = pd.MultiIndex.from_arrays(independent,
                                          names=self.model.independent)
        result = pd.DataFrame(model_output,
                              index=index,
                              columns=self.model._ordered_var['algebraic'])
        return result

    def _solve_algebraic_lsa(self, alg_function, parameters, *args, **kwargs):
        """
        """
        model_output = self._solve_algebraic_generic(alg_function, *args,
                                                     **kwargs)

        index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                          names=self.model.independent)

        columns = pd.MultiIndex.from_tuples(list(product(
                        self.model._ordered_var['algebraic'], parameters)))
                        #, sortorder=0)

        indep_len = len(self.model._independent_values.values()[0])

        result = pd.DataFrame(model_output.reshape(indep_len, -1),
                              index=index, columns=columns)

        return result

    def solve(self, *args, **kwargs):
        """
        Calculate the algebraic equations in function of the independent values

        Returns
        -------
        result : pd.DataFrame
            Contains all outputs from the algebraic equation in function of the
            independent values
        """
        return self._solve_algebraic(*args, **kwargs)


class HybridSolver(OdeSolver, AlgebraicSolver):
    """
    Class for solving hybrid system of odes and algebraic equations by using.
    """
    def solve(self, procedure="odeint", **kwargs):
        """
        Solve hybrid system of odes and algebraic equations. After calculating
        the odes, the algebraic equations are calculated again.

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from both odes and algebraics
        """
        self._initial_conditions = [self.model.initial_conditions[var]
                                    for var in self.model._ordered_var['ode']]
        self._check_ode_integrator_setting(procedure)
        # Solving ODEs
        ode_output, xdata = self._ode_procedure[procedure](self.model.fun_ode,
                                                           **kwargs)

        ode_result = pd.DataFrame(ode_output, index=xdata,
                                  columns=self.model._ordered_var['ode'])
        alg_result = self._solve_algebraic(ode_result.values, **kwargs)

        result = pd.concat([ode_result, alg_result], axis=1)

        return result
