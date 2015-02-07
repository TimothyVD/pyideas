# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:03:10 2015

@author: timothy
"""
import numpy as np

class oed(object):
    """
    """

    def __init__(self, confidence):
        """
        """
        self.confidence = confidence

    @staticmethod
    def A_criterion(FIM):
        '''OED design A criterion
        With this criterion, the trace of the inverse of the FIM is minimized,
        which is equivalent to minimizing the sum of the variances of the
        parameter estimates. In other words, this criterion minimizes the
        arithmetic average of the variances of the parameter estimate.
        Because this criterion is based on an inversion of the FIM,
        numerical problems will arise when the FIM is close to singular.
        '''
        return np.linalg.inv(FIM).trace()

    @staticmethod
    def modA_criterion(FIM):
        '''OED design modified A criterion
        With this criterion, the trace of the inverse of the FIM is minimized,
        which is equivalent to minimizing the sum of the variances of the
        parameter estimates. In other words, this criterion minimizes the
        arithmetic average of the variances of the parameter estimate.
        Because this criterion is based on an inversion of the FIM,
        numerical problems will arise when the FIM is close to singular.
        '''
        return FIM.trace()

    @staticmethod
    def D_criterion(FIM):
        '''OED design D criterion
        Here, the idea is to maximize the determinant of the FIM
        (Box and Lucas, 1959). The latter is inversely proportional to the
        volume of the confidence region of the parameter es- timates, and this
        volume is thus minimized when maximizing det (FIM). In other words,
        one minimizes the geometric average of the variances of the parameter
        estimates. More- over, D-optimal experiments possess the property of
        being invariant with respect to any rescaling of the parameters
        (Petersen, 2000; Seber and Wild, 1989). According to Walter and
        Pronzato (1997), the D-optimal design criterion is the most used
        criterion. However, several authors have pointed out that this
        criterion tends to give excessive importance to the parameter which
        is most influential.
        '''
        return np.linalg.det(FIM)

    @staticmethod
    def E_criterion(FIM):
        '''OED design E criterion
        The E-optimal design criterion maximizes the smallest eigenvalue of
        the FIM and thereby minimizes the length of the largest axis of the
        confidence ellipsoid. Thus, these designs aim at minimizing the
        largest parameter estimation variance and thereby at maximizing the
        distance from the singular, unidentifiable case.
        '''
        return np.min(np.linalg.eigvals(FIM))

    @staticmethod
    def modE_criterion(FIM):
        '''OED design modE criterion
        With this criterion, the focus is on the minimization of the condition
        number, which is the ratio between the largest and the smallest
        eigenvalue, or, in other words, the ratio of the shortest and the
        longest ellipsoid axes. The minimum of this ratio is one, which
        corresponds to the case where the shape of the confidence ellipsoid
        is a (hyper)sphere.
        '''
        w = np.linalg.eigvals(FIM)
        return np.max(w)/np.min(w)

