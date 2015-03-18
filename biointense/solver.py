# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 11:52:12 2015

@author: timothy
"""
from __future__ import division

from scipy.integrate import odeint, ode
from itertools import product

try:
    import odespy
except:
    pass

import pandas as pd


ODE_INTEGRATORS = {}
ODE_INTEGRATORS['ode'] = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
ODE_INTEGRATORS['odespy'] = odespy.list_available_solvers()

STD_ODE_INTEGRATOR = {'ode': 'lsoda', 'odespy': 'lsoda_scipy', 'odeint': ''}


class Solver(object):

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


class OdeSolver(Solver):
    """
    """
    def __init__(self, model, ode_solver_options=None, ode_integrator=None):
        """
        """
        self.model = model
        self.ode_solver_options = ode_solver_options or {}
        self.ode_integrator = ode_integrator
        self._initial_conditions = [self.model.initial_conditions[var]
                                    for var in self.model._ordered_var['ode']]
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

    def _solve_odeint(self):
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
        res = odeint(self.model.fun_ode,
                     self._initial_conditions,
                     self.model._independent_values.values()[0],
                     args=(self.model.parameters,),
                     **self.ode_solver_options)
        # Put output in pandas dataframe
        result = pd.DataFrame(res, index=self.model._independent_values.values()[0],
                              columns=self.model._ordered_var['ode'])

        return result

    def _solve_ode(self):
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
        # Make wrapper function to
        def wrapper(independent_values, initial_conditions, parameters):
            return self.model.fun_ode(
                initial_conditions, independent_values, parameters)

        solver = ode(wrapper).set_integrator(self.ode_integrator,
                                             **self.ode_solver_options)

        solver.set_initial_value(self._initial_conditions,
                                 self.model._independent_values.values()[0][0])
        solver.set_f_params(self.model.parameters)

        xdata = self.model._independent_values.values()[0]
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

        result = pd.DataFrame(model_output, index=xdata,
                              columns=self.model._ordered_var['ode'])

        return result

    def _solve_odespy(self):
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
        solver = odespy.__getattribute__(self.ode_integrator)
        solver = solver(self.model.fun_ode)
        if self.ode_solver_options is not None:
            solver.set(**self.ode_solver_options)
        solver.set_initial_condition(self._initial_conditions)
        solver.set(f_args=(self.model.parameters,))

        model_output, xdata = solver.solve(self.model._independent_values.values()[0])

        result = pd.DataFrame(model_output, index=xdata,
                              columns=self.model._ordered_var['ode'])

        return result

    def solve(self, procedure='odeint'):
        """
        Calculate the ode equations using scipy integrate odeint solvers

        Returns
        -------
        result : pd.DataFrame
            Contains all outputs from the ode equations in function of the
            independent values
        """
        self._check_ode_integrator_setting(procedure)

        return self._ode_procedure[procedure]()


class AlgebraicSolver(Solver):
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

        index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                          names=self.model.independent)
        result = pd.DataFrame(model_output,
                              index=index,
                              columns=self.model._ordered_var['algebraic'])
        return result

    def _solve_algebraic_lsa(self, alg_function, *args, **kwargs):
        """
        """
        model_output = self._solve_algebraic_generic(alg_function, *args,
                                                     **kwargs)

        index = pd.MultiIndex.from_arrays(self.model._independent_values.values(),
                                          names=self.model.independent)

        columns = pd.MultiIndex.from_tuples(list(product(
                        self.model._ordered_var['algebraic'],
                        self.model.parameters.keys())))

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
    def solve(self, procedure="odeint"):
        """
        Solve hybrid system of odes and algebraic equations. After calculating
        the odes, the algebraic equations are calculated again.

        Returns
        -------
        result : pd.DataFrame
        Contains all outputs from both odes and algebraics
        """
        self._check_ode_integrator_setting(procedure)
        ode_result = self._ode_procedure[procedure]()
        alg_result = self._solve_algebraic(ode_result.values)

        result = pd.concat([ode_result, alg_result], axis=1)

        return result
