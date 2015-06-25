# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 20:11:00 2015

@author: joris
"""

from __future__ import division
import os
import unittest
import nose

from pandas.util.testing import assert_almost_equal
import biointense

def test_LSA():
    output = {}
    # set working directory on super folder
    execfile(str(os.path.join(biointense.BASE_DIR, "..", "examples", "LSA_alg_ode.py")),
	     output)
    
    dir_sens, num_sens = output['LSA_comparison']()

    assert_almost_equal(dir_sens, num_sens)

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
