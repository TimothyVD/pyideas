.. BioIntense documentation master file, created by
   sphinx-quickstart on Mon Apr 08 16:37:40 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BioIntense's documentation!
======================================

Contents:

.. toctree::
   :maxdepth: 2


This biointense model environment is an object oriented python implementation for model building and analysis, 
focussing on sensitivity and identifiability analysis. 

ODE-Generator
---------------     
ODE-generator class is a simplified version of a ODE creation environment, together with functions for 
sensitivity and distinguishability/identifiability analysis

.. autoclass:: ode_generator.odegenerator
   :members: 

OED
-----
OED class gives the functionalities for FIM-based experimental design for parameter estimation.

.. autoclass:: optimalexperimentaldesign.OED
   :members:   

   
Support Functions
-------------------

.. autofunction:: plotfunctions.scatterplot_matrix




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

