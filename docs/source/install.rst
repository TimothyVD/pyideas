How to install pyIDEAS?
========================

The dependencies of pyIDEAS are basic packages of the scientific python ecosystem:

* `NumPy <http://www.numpy.org>`__
* `pandas <http://pandas.pydata.org/>`__
* `SymPy <http://www.sympy.org/en/index.html>`__
* `SciPy <http://www.scipy.org>`__
* `matplotlib <http://matplotlib.sourceforge.net/>`__

Further, optional dependencies for specific functionality include:

* `odespy <http://hplgit.github.io/odespy/doc/api/odespy.html>`__: more ODE solvers
    * download and install: https://github.com/hplgit/odespy
    * for the use of fortran librairies, install gfortran package (debian based systems)
* `inspyred <http://pythonhosted.org//inspyred/>`__: global optimization + robust OED

Development requirement for testing:

* `py.test <http://pytest.org/latest/>`__ >= 2.4
