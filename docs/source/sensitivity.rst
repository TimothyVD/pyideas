Sensitivity classes
====================
Local Sensitivity
------------------
Two different ways to calculate the local sensitivity for a set of parameters
are provided. First, the direct local sensitivity uses the system equations of
the Model/AlgebraicModel classes to derive the appropriate matrices which can 
be used to calculate the local sensitivity. Second, the numerical local
sensitivity performs perturbations (= small changes) of the values of the
parameters of interest. By comparing the effect of such a small change on the
model output, the sensitivity can be assessed. The advantage of the numerical
local sensitivity, is that this class can also be applied to models which are
externally provided by a script. In this way, it allows calculation of the
sensitivity, optimisation, (robust) OED for models not written in the
framework.

Direct Local Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: biointense.DirectLocalSensitivity
   :members: 

Numerical Local Sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: biointense.NumericalLocalSensitivity
   :members: 
  
Global Sensitivity
-------------------
This class will provide a coupling with the pySTAN package to allow this kind
of calculations.
