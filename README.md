biointense
==========

Biointense FP7 WP9

Authors: Timothy Van Daele, Stijn Van Hoey, Daan Van Hauwermeiren, Joris Van den Bossche

TO DO
-----
* Unit testing
* SBML support (export and import)
* Add ABC package (MCMC) (http://www.theosysbio.bio.ic.ac.uk/resources/abc-sysbio/Background/)
* Use ode_FIM directly on odegenerator
* Make input structure of model definition more generic
	* Parameters (theta)
	* Dependent parameters (theta_dep)
	* Inputs (like stepfunction/external function) (u)
	* ODEs (dx/dt = f(theta, theta_dep, u)
	* Algebraic equations g(x, theta, theta_dep, u)
* More logical names for different components
	* modelmaker
	* modelrunner
	* measurements
	* modeloptimizer
	* modelFIM
* Make calculations with only Algebraic equation easier (allow multiple undependent variables)
* Allow data to be added for multiple experiments (using multi-index or ....) so that all those experiments can be used at once to optimize/fit the model
* Make one function calcLSA to switch between numeric and analytical function
* Rename output numerical and analytical sensitivity
* Add algebraic sensitivity to calculation directly
* See if algebraic sensitivity can be calculated more efficiently
* Add documentation to every function
* Make bioinspyred optimization more generic
* Automatically write out the version of biointense which was made to produce results
* Fix/Update setup.py file
* Add raise Exceptions and more check functions
	* Check input of model (every symbol has at least 2 letters)
	* ...
* Replace write to file with append to string and afterwards evaluate
* Move model definition functionality from odegenerator to odemaker (odegenerator becomes DAErunner)
* Better coupling with pySTAN
* Check how current dense matrix operations can be replaced by sparse ones (especially important for ODE system with more than 1000 ODEs)
* Allow biointense to do calculations with PDEs
* Add log file functionality
* Perform Optimization and FIM calculation for multiple experiments at the same time
* ...


Functionalities
---------------
* ODE definition and run
* LSA (both analytical and numerical)
* Check mass balance
* Quasi steady-state generator
* Pulse- and stepfunction
* Measurements
* Optimization (both local and global (bioinspyred))
* Fisher Information Matrix (FIM)
	* Different optimization criteria (A, modA, D, E, modE)
	* Parameter/Model output confidence and correlation

Functionality to be added:

Short term
----------
* LSA for variables not only parameters => DONE
* 

Long term
----------
* Add identifiability tools
* 


