
## License: LELIJKE DASHONDEN
## All rights to us and none to the others!



class Solver(object):
    
    def __init__(functie):
        """
       
        odeint(functie, init, tijd, args*)
        """
    
    def _check_solver_sanity():
        """
        check whether solver setting is compatible with system
        check the external event: can the solver cope with the time step of 
        the measurement?
        """

class odeint(Solver):
    """
    """
    def __init__():
        """
        """
        
# imports
import warnings
        
        
class BaseModel(object):

    def __init__(system, name, parameters):
        """

        ODE equations are defined by the pre-defined d, whereas algebraic
        equations are defined by equations without a d in front

        Example
        --------
        >>> sir = {'dS' : '-k*I*B/N',
                      'dI' : 'k*I*B/N - gam*I*t',
                      'dR' : 'gam*I',
                      'N' : 'S + I + R + NA'}
        >>> param = {'k': 2., 'gam' : 0.3}
        >>> name = 'SIR1'
        to be adjusted: >>> S_event = f(t)
        >>>
        """
        
        self.variables = {'algebraic': [], 
                          'ode': [], 
                          'event': [],
                          'independent':[]
                          }
        
        # solver coomunication
        self.independent_values = None
        self.systemfunctions = {'algebraic' : {}, 'ode' : {}}
        self.parameters = None
        self.initial_conditions = None

                       
        self.variables_of_interest = []        
        self._initial_up_to_date = False

    def _parse_system_string(self, system, parameters):
        """
        split the system in ODE & algebraic
        extract variable names
        first letter == d ==> to ODE
        else ==> to algebraic
        extract info from system and parameters and store them into the attributes
        """
        # assert that 'parameters' and 'system' are a dict
        if not isinstance(parameters, dict):
            raise typeError("parameters is not a dict")
        if not isinstance(system, dict):
            raise typeError("system is not a dict")
        # store the parameter
        self.parameters = parameters
        # extract system information
        # loop over the dictionairy: system
        for key, value in system.iteritems():
            # if first letter == d, equation is ODE
            if key[0] == "d":
                #get rid of the first letter, d
                self.systemfunctions['ode'][key[1:]] = value
                self.variables['ode'].append(key[1:])
            else:
                self.systemfunctions['algebraic'][key] = value
                self.variables['algebraic'].append(key)
    
    def __str__():
        """
        """

    def __repr__():
        """
        """

    def _check_system(self):
        """
        check sys ...
        define ODE, algebraic, pde vars seperately
        if in sys && not in parameter:
            WARNING: another var found
        check if parameter of choice and independent vars are found in the
        system/ variable list.
        """

    def _check_name(self):
        """
        """
        if not isinstance(str):
            raise TypeError("model name is not a string")


    def _check_parameters(self):
        """
        check is type == dict
        check for a match between system and parameters

        see: _checkParinODEandAlg(self): in ode_generator.py
        """

    @classmethod
    def from_external(cls, ext_sys):
        """
        initialise system from external function
        integratei met andere paketten om het in een 
        """
        return cls(None)

    def set_independent(self, independentVar):
        """
        set independent variable, mostly time
        1D
        """
        # check the data type of the input
        if not isinstance(independentVar, str):
            raise TypeError("Independent variable is not given as a string")
        # check if independent variable is not already implemented
        if self.variables['independent']:
            warnings.warn("Warning: independent variable is already given. "
                           + "Overwriting original " 
                           + self.variables['independent'] + " with " 
                           + independentVar)
        # setting the new independent variable
        self.variables['independent'].append(independentVar)

    def set_initial(self):
        """
        set initial conditions
        """

    def set_variables_of_interest(self, variables_of_interest):
        """
        set the variables to be exported to the output
        """
        if self.variables_of_interest:
            warnings.warn("Warning: variables of interest are already given. "
                           + "Overwriting original variables.")
        # test if the input is a list
        if isinstance(variables_of_interest, list):
            for element in variables_of_interest:
                # if the input is a list, check if all inputs are strings                
                if not isinstance(element, str):
                    raise TypeError("Elements in list are not strings")
            self.variables_of_interest = variables_of_interest
            return
        # test if the input is a string
        if isinstance(variables_of_interest, str):
            self.variables_of_interest = [variables_of_interest]
            return
        # if the input is no string nor list of strings, raise error
        raise TypeError("Input is not a string nor a list of strings")

    def get_summary(self):
        """
        returns summary of the model
            parameters
            variables & type
            events
            time steps (init defined user, event, measurements)
            initial conditions
            ready to run!
        """
        
    def _check_for_init(self):
        """
        """

    def _check_for_independent(self):
        """
        """
        
    def add_event(variable, ext_fun, tijdsbehandeling, idname):
        """
        Variable is defined by external influence. This can be either a
        measured value of input (e.g. rainfall) or a function that defines
        a variable in function of time

        See also:
        ---------
        functionMaker
        
        plug to different files: step input ...
        + add control to check whether external function addition is possible
        
        + check if var exists in ODE/algebraic => make aggregation function to
        contacate them.
        """
        self._initial_up_to_date = False
        
        return True

    def list_current_events(self):
        """
        """
        return NotImplementedError

    def exclude_event(self, idname):
        """
        """        
        return NotImplementedError

    def _collect_time_steps(self):
        """
        """
        return NotImplementedError
        
    def initialize_model(self):
        """
        make string object of model (templating based would be preferred)
        adjust variables to cope with events:
            S & S_event ==> aggregation function
        make Solver object
        set verbose option
        """
        _collect_time_steps(_fromuser, _fromevents, _frommeasurements)
        Solver(integrate option)

    def run():
        """
        generate dataframe
        """
        if not self._initial_up_to_date:
            self.initialize_model
            
    def plot(self):
        """
        plot dataframe
        """
        return NotImplementedError

    def save(self):
        """
        Saves the object to a file, cfr. pickle
        """
        return NotImplementedError

    @classmethod
    def load(cls):
        """
        Loads the object from a file, cfr. pickle
        """
        return cls(None)

    def export_to(environment):
        """
        Converts the model to be used in other environment

        Parameters
        -----------
        environment : matlab, openModelica, libSBML
        """
        return NotImplementedError



def check_mass_balance():
    """
    Check the mass balance of the model.

    This method calls the external utility _getCoefficients
    """
    return True


class AlgebraicModel(BaseModel):

    def __init__():
        """
        with multidimensional independant variables
        """

    def set_independent(self):
        """
        set independent variable, mostly time
        """


class ReactionModel(BaseModel):

    def __init__():
        """
        """

    @classmethod
    def from_diagram(cls):
        """
        Creates model based on the
        """


class EnzymaticModel(ReactionModel):

    def __init__():
        """
        """

    def _getCoefficients(self):
        """
        """

    @classmethod
    def make_quasi_steady_state(cls):
        """
        Converts the ODE system to the Quasi Steady State version
        
        Combines the old versions make_QSSA and QSSAtoModel to create QSSA
        model based on a defined ODE system.
        """
        return True

    
