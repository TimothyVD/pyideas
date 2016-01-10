Biointense Model Classes
=========================
Two types of models are available at this moment: First, the Model class, which
provides an easy interface to write sets of ODEs combined with Algebraic 
equations. The AlgebraicModel class provides an interface in the case NO ODEs 
are involved. The advantage of the AlgebraicModel class is that for this class,
it is possible to provide multiple independent variables. These model classes 
can afterwards be used to perform sensitivity analyses, optimisation, (robust)
Optimal Experimental Design,... 

Model
------
.. autoclass:: biointense.Model
   :members: 

AlgebraicModel
---------------
.. autoclass:: biointense.AlgebraicModel
   :members: 
  

   
