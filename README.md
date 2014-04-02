biointense
==========

Biointense FP7 WP9

Authors: Timothy Van Daele, Stijn Van Hoey

TO DO
-----
* Unit testing
* SBML support (export and import)
* Use ode_FIM directly on odegenerator
* Make input structure of model definition more generic
	* Parameters (theta)
	* Dependent parameters (theta_dep)
	* Inputs (like stepfunction/external function) (u)
	* ODEs (dx/dt = f(theta, theta_dep, u)
	* Algebraic equations g(x, theta, theta_dep, u)
* More logical names for different components
	* odegenerator
	* daerunner
	* measurements
	* daeoptimizer
	* daeFIM
* Make one function calcLSA to switch between numeric and analytical function
* Rename output numerical and analytical sensitivity
* Add algebraic sensitivity to calculation directly
* See if algebraic sensitivity can be calculated more efficiently
* Add documentation to every function
* Add documentation within every function
* Make bioinspyred optimization more generic
* Automatically write out the version of biointense which was made to produce results
* Fix/Update setup.py file
* Add raise Exceptions and more check functions
	* Check input of model (every symbol has at least 2 letters)
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


