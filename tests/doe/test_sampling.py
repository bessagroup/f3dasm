##############################################################
## To test the F3DASM, install the path to the package      ##
## to the local python envionment as follows:               ##
## pip install -e .                                         ##
##############################################################

import unittest
import numpy
from f3dasm.doe import sampling
from f3dasm.doe.sampling import Sobol, Linear, SamplingMethod


RANGE_VARS = {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]} 
BAD_RANGE = RANGE_VARS
BAD_RANGE['F22'] = [-0.15, 1, 100]
SAMPLE_SIZE = 5


class TestSamplingMethod(unittest.TestCase):

    def test_validate_rante(self):
        """Test that method raises a typeError when values pass to the sampling method contatin a list with more than 2 elements"""
        self.sobol_with_bad_range = Sobol(SAMPLE_SIZE, BAD_RANGE)
        self.assertRaises(TypeError, self.sobol_with_bad_range.validate_range())

    def test_get_dimensions(self):
        """test that method returs the correct number of dimensions"""
        self.sobol = Sobol(SAMPLE_SIZE, RANGE_VARS)
        self.assertEqual(self.sobol.compute_dimensions(), 3)

    
class TestSobolSampling(unittest.TestCase):

    def setUp(self) -> None:
        """set up test fixtures"""
        self.sobol_sampling = Sobol(SAMPLE_SIZE, RANGE_VARS)
        self.sampling_output = self.sobol_sampling.compute_sampling()

    def test_output_type(self):
        """Test that output of sampling method is an array"""
        self.assertIsInstance(self.sampling_output, numpy.ndarray)

    def test_output_dimentions(self):
        """test that sampling method produces an array with the correct dimensions"""
        self.assertEqual(self.sampling_output.shape[0], SAMPLE_SIZE )
        self.assertEqual(self.sampling_output.shape[1], 3 )


# class TestLinearSampling(unittest.TestCase):

#     def setUp(self) -> None:
#         """set up test fixtures"""
#         self.sobol_sampling = LinearSampling(SAMPLE_SIZE, RANGE_VARS)

#     def test_output_type(self):
#         """Test that output of sampling method is an array"""
#         self.assertIsInstance(self.sobol_sampling.get_sampling(), numpy.ndarray)

#     def test_get_dimensions(self):
#         """test that method returs the correct number of dimensions"""
#         self.assertEqual(get_dimensions(RANGE_VARS), 3)

#     def test_get_sampling(self):
#         """test that sampling method produces an array with the correct dimensions"""

#         self.assertEqual(self.sobol_sampling.get_sampling().shape[0], SAMPLE_SIZE )
#         self.assertEqual(self.sobol_sampling.get_sampling().shape[1], 3 )

if __name__ == '__main__':
    unittest.main()