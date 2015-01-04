# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 20:11:00 2015

@author: joris
"""

from __future__ import division
import os
import unittest
import nose

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_almost_equal

from collections import OrderedDict

# set working directory on super folder
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(file_dir, '..'))


class TestExample_modsim(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import modsim
        output = modsim.run_modsim_models()
        cls.M1, cls.M2, cls.M3, cls.FIM1, cls.FIM2, cls.FIM3 = output

    def test_model1(self):

        # test the optimized parameters
        expected = OrderedDict([('W0', 2.080398434017849),
                                ('Wf', 9.7522237047714704),
                                ('mu', 0.065925117843826847)])
        result = self.M1.Parameters
        self.assertEqual(result, expected)

        # test the parameter confidence
        expected = pd.DataFrame(
            {'delta': [2.4199818502274191, 2.9538718930120691, 0.068507321248180181],
             'lower': [-0.33958341620957011, 6.7983518117594013, -0.0025822034043533337],
             'percent': [116.32299903022599, 30.289213849420086, 103.91687339940816],
             'significant': [0.0, 1.0, 0.0],
             't_reference': [2.7764451051977987, 2.7764451051977987, 2.7764451051977987],
             't_value': [2.3868410618233393, 9.166448224772779, 2.6717943047867063],
             'upper': [4.500380284245268, 12.706095597783539, 0.13443243909200703],
             'value': [2.080398434017849, 9.7522237047714704, 0.065925117843826847]},
            index=['W0', 'Wf', 'mu'],
            columns=['value', 'lower', 'upper', 'delta', 'percent', 't_value', 't_reference', 'significant'])
        result = self.FIM1.get_parameter_confidence()
        assert_frame_equal(result, expected)

        # test the parameter correlation
        expected = np.array([0.44630711, -0.84803599, -0.74509016])
        result = self.FIM1.get_parameter_correlation().values[[1,2,2], [0,0,1]]
        assert_almost_equal(result, expected)

    def test_model2(self):

        # test the optimized parameters
        expected = OrderedDict([('Wf', 10.718908267877257),
                                ('mu', 0.03095203722848679)])
        result = self.M2.Parameters
        self.assertEqual(result, expected)

        # test the parameter confidence
        expected = pd.DataFrame(
            {'delta': [4.4406191136459041, 0.029570974121414056],
             'lower': [6.278289154231353, 0.0013810631070727344],
             'percent': [41.42790480774692, 95.538054258342413],
             'significant': [1.0, 1.0],
             't_reference': [2.570581836614739, 2.570581836614739],
             't_value': [6.2049525520152455, 2.690636580962475],
             'upper': [15.159527381523162, 0.060523011349900846],
             'value': [10.718908267877257, 0.03095203722848679]},
            index=['Wf', 'mu'],
            columns=['value', 'lower', 'upper', 'delta', 'percent', 't_value', 't_reference', 'significant'])
        result = self.FIM2.get_parameter_confidence()
        assert_frame_equal(result, expected)

        # test the parameter correlation
        expected = np.array([-0.9469973])
        result = self.FIM2.get_parameter_correlation().values[[1], [0]]
        assert_almost_equal(result, expected)

    def test_model3(self):

        # test the optimized parameters
        expected = OrderedDict([('D', 0.041087536998484858),
                                ('W0', 2.0423292965130431),
                                ('mu', 0.066868735215877745)])

        result = self.M3.Parameters
        self.assertEqual(result, expected)

        # test the parameter confidence
        expected = pd.DataFrame(
            {'delta': [0.050239739651197796, 2.5948441932397963, 0.11182971611492232],
             'lower': [-0.0091522026527129374, -0.55251489672675325, -0.044960980899044573],
             'percent': [122.27488752380175, 127.05317392597198, 167.23767206586666],
             'significant': [0.0, 0.0, 0.0],
             't_reference': [2.7764451051977987, 2.7764451051977987, 2.7764451051977987],
             't_value': [2.2706584822311466, 2.1852622956239611, 1.6601792352767826],
             'upper': [0.091327276649682654, 4.6371734897528398, 0.17869845133080006],
             'value': [0.041087536998484858, 2.0423292965130431, 0.066868735215877745]},
            index=['D', 'W0', 'mu'],
            columns=['value', 'lower', 'upper', 'delta', 'percent', 't_value', 't_reference', 'significant'])
        result = self.FIM3.get_parameter_confidence()
        assert_frame_equal(result, expected)

        # test the parameter correlation
        expected = np.array([-0.73682127,  0.92973434, -0.93054158])
        result = self.FIM3.get_parameter_correlation().values[[1,2,2], [0,0,1]]
        assert_almost_equal(result, expected)


if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
