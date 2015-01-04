# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 11:52:12 2015

@author: timothy
"""
from __future__ import division

from scipy.integrate import odeint, ode

try:
    import odespy
except:
    pass

import pandas as pd


class Solver(object):

    def __init__(self, model):
        """
        odeint(functie, init, tijd, args*)
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


class BaseOdeSolver(Solver):
    """
    """
    def __init__(self, model, ode_solver_options=None, ode_integrator=None):
        """
        """
        self.model = model
        self.ode_solver_options = ode_solver_options or {}
        self.ode_integrator = ode_integrator

    def solve(self):
        """
        """


class OdeintSolver(BaseOdeSolver):
    """
    """

    def _solve_odeint(self):
        """
        """
        res = odeint(self.model.systemfunctions['ode'],
                     self.model.initial_conditions,
                     self.model.independent_values,
                     args=(self.model.parameters,),
                     **self.ode_solver_options)
        # Put output in pandas dataframe
        result = pd.DataFrame(res, index=self.model.independent_values,
                              columns=self.model.variables['ode'])

        return result

    def solve(self):
        """
        """
        return self._solve_odeint()


class OdeSolver(BaseOdeSolver):
    """
    """
    def _solve_ode(self):
        """
        """
        # Default value for odespy
        self.ode_integrator = self.ode_integrator or 'lsoda'
        initial_conditions = [self.model.initial_conditions[var] for var in self.model.variables['ode']]

        def wrapper(independent_values, initial_conditions, parameters):
            return self.model.systemfunctions['ode'](
                initial_conditions, independent_values, parameters)

        solver = ode(wrapper).set_integrator(self.ode_integrator,
                                        **self.ode_solver_options)

        solver.set_initial_value(initial_conditions,
                            self.model.independent_values[0])
        solver.set_f_params(self.model.parameters)

        xdata = self.model.independent_values
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
                              columns=self.model.variables['ode'])

        return result

    def solve(self):
        """
        """
        return self._solve_ode()


class OdespySolver(BaseOdeSolver):
    """
    """
    def _solve_odespy(self):
        """
        """
        # Default value for odespy
        self.ode_integrator = self.ode_integrator or 'lsoda_scipy'

        solver = odespy.__getattribute__(self.ode_integrator)
        solver = solver(self.model.systemfunctions['ode'])
        if self.ode_solver_options is not None:
            solver.set(**self.ode_solver_options)
        solver.set_initial_condition(self.model.initial_conditions)
        solver.set(f_args=(self.model.parameters,))

        model_output, xdata = solver.solve(self.model.independent_values)

        result = pd.DataFrame(model_output, index=xdata,
                              columns=self.model.variables['ode'])

        return result

    def solve(self):
        """
        """
        return self._solve_odespy()


class AlgebraicSolver(Solver):
    """
    """
    def _solve_algebraic(self):
        """
        """
        alg_function = self.model.systemfunctions['algebraic']
        model_output = alg_function(self.model.independent_values,
                                    self.model.parameters)

        result = pd.DataFrame(model_output, index=self.model.independent_values,
                              columns=self.model.variables['algebraic'])

        return result

    def solve(self):
        """
        """
        return self._solve_algebraic()


class HybridOdeintSolver(OdeintSolver, AlgebraicSolver):
    """
    """
    def solve(self):
        """
        """
        ode_result = self._solve_odeint()
        alg_result = self._solve_algebraic()

        result = pd.concat(ode_result, alg_result, axis=1)

        return result


class HybridOdeSolver(OdeSolver, AlgebraicSolver):
    """
    """
    def solve(self):
        """
        """
        ode_result = self._solve_ode()
        alg_result = self._solve_algebraic()

        result = pd.concat(ode_result, alg_result, axis=1)

        return result


class HybridOdespySolver(OdespySolver, AlgebraicSolver):
    """
    """
    def solve(self):
        """
        """
        ode_result = self._solve_odespy()
        alg_result = self._solve_algebraic()

        result = pd.concat(ode_result, alg_result, axis=1)

        return result

