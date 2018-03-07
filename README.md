pyideas
==========

pyideas is a python package for (simple) model optimisation, confidence calculations, and optimal experimental design.

Most useful information can currently be found in examples. If you would have issues, please use the issues tool, not sure whether I will have time, but feel free to improve the package :-)

Very short overview of current package capabilities:
Models
------
* Easy model definitions for simple models
	* Algebraic models with multiple independents (i.e. without any ODEs) - AlgebraicModel
	* Combined Algebraic and ODE models - Model
* Capability to call external MATLAB models, and use in pyideas workstream

Local Sensitivity Analysis (LSA)
---------------------------------
* Analytical derivation of sensitivity functions for simple models - DirectLocalSensitivity
* Numerical local sensitivity for all models - NumericalLocalSensivity
* Usage of different sensitivity measure:
	* Absolute Sensitivity (AS)
	* Parameter Relative Sensitivity (PRS)
	* Total Relative Sensitivity (TRS)
	
Parameter Optimisation
----------------------
* Parameter optimisation using subset of model parameters
* Define measurements using pandas DataFrame
* Availability of different optimisation functions

Parameter and model confidence
-----------------------------
* Using Fisher Information Matrix (FIM), which is based on LSA, to estimate covariance-variance matrix

Optimal Experimental Design
----------------------------
* For simple models, the FIM can be evaluated for all potential experimental conditions - BruteOED
* Robust Optimal Experimental Design (AKA maximin design, see paper Asprey) takes into account - RobustOED
	* Uncertainty of current parameter estimates
	* Information available in a set of experimental conditions

License
--------
Copyright (c) [2018] [Timothy Van Daele]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
