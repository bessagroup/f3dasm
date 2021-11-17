##############################################################
## To test the F3DASM, install the path to the package      ##
## to the local python envionment as follows:               ##
## pip install -e .                                         ##
##############################################################

import unittest
import numpy
from f3dasm.doe import sampling
from f3dasm.doe.sampling import Sobol, Linear, samples_to_dict, validate_range, SamplingMethod


DOE_VARS = {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1], 'radius': [0.3, 5],
            'material1': {'STEEL': {'E': [0,100], 'u': {0.1, 0.2, 0.3} }, 
                    'CARBON': {'E': 5, 'u': 0.5, 's': 0.1 } },
            'material2': { 'CARBON': {'x': 2} }} 
TOP_RANGES = {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1], 'radius': [0.3, 5]}
BAD_RANGE = [-0.15, 1, 100]
SAMPLE_SIZE = 5
LIST_RANGE = [5, 10]
SAMPLES = numpy.array([ [-0.15,    -0.1, -0.2, 0.3], 
            [ 0.425,    0.025,    0.4,      2.65   ],
            [ 0.7125,  -0.0375,   0.1,      1.475  ],
            [ 0.1375,   0.0875,   0.7,      3.825  ],
            [ 0.28125, -0.00625,  0.55,     4.4125 ]])

#TODO: update tests to match new definition of sampling methods

class TestValidateRange(unittest.TestCase):

    def test_returns_true(self):
        """Test that method validate_range returns true
        range of values is a list with 2 numeric elements"""
        self.assertTrue(validate_range(LIST_RANGE))


class TestSamplesToDict(unittest.TestCase):

    def test_output_dimensions(self):
        """Test output dictinary contains same number of elemenets as columns in samples array"""
        self.dictionary = samples_to_dict(SAMPLES, TOP_RANGES.keys())
        self.assertEqual(len(self.dictionary.keys()), SAMPLES.shape[1] )


class TestSamplingMethod(unittest.TestCase):

    def setUp(self):
        """set up test fixtures"""
        self.method_instance = Sobol(SAMPLE_SIZE, DOE_VARS)
        
    def test_select_values_for_sampling(self):
        """Test that method selects top elements in a dictionary that contain a valid range"""
        self.selected_values = self.method_instance.sampling_ranges
        self.assertEqual(len(self.selected_values.keys()), 4)

    def test_select_fixed_values(self):
        """Test that method selects top elements in a dictionary which don't contain valid range"""
        self.fixed_values = self.method_instance.select_fixed_values()
        self.assertEqual(len(self.fixed_values.keys()), 2)

    def test_create_combinations(self):
        """Test fixe and sampled variables are combine correctly"""
        # TODO
        pass


class TestSobolSampling(unittest.TestCase):

    def setUp(self) -> None:
        """set up test fixtures"""
        self.sobol_sampling_dict = Sobol(SAMPLE_SIZE, DOE_VARS)
        self.sampling_output = self.sobol_sampling_dict.compute_sampling()
        
    def test_output_type_array(self):
        """Test that output of sampling method is an array"""
        self.assertIsInstance(self.sampling_output, numpy.ndarray)

    def test_sampling_are_floats(self):
        """Test that sampled values are of type float when aprox='float"""
        self.assertEqual(self.sampling_output.dtype, 'float64')

    def test_sampling_are_int(self):
        """Test that sampled values are of type integers when aprox='int"""
        # TODO: dtypes: int8 or int16
        pass

class TestSobolLinear(unittest.TestCase):

    def setUp(self) -> None:
        """set up test fixtures"""
        self.linear_sampling_dict = Sobol(SAMPLE_SIZE, DOE_VARS)
        self.sampling_output = self.linear_sampling_dict.compute_sampling()
        
    def test_output_type_array(self):
        """Test that output of sampling method is an array"""
        self.assertIsInstance(self.sampling_output, numpy.ndarray)

    def test_sampling_are_floats(self):
        """Test that sampled values are of type float when aprox='float"""
        self.assertEqual(self.sampling_output.dtype, 'float64')

    def test_sampling_are_int(self):
        """Test that sampled values are of type integers when aprox='int"""
        # TODO: dtypes: int8 or int16
        pass




# class TestLinearSampling(unittest.TestCase):

#     def setUp(self) -> None:
#         """set up test fixtures"""
#         self.linear_sampling = Linear(SAMPLE_SIZE, RANGE_VARS)
#         self.sampling_output_dict = self.linear_sampling.compute_sampling()
#         self.linear_sampling_list = Linear(SAMPLE_SIZE, LIST_RANGE)
#         self.sampling_output_list = self.linear_sampling_list.compute_sampling()

#     def test_output_type_dict(self):
#         """Test that output of sampling method is an array"""
#         self.assertIsInstance(self.sampling_output_dict, numpy.ndarray)

#     def test_output_type_list(self):
#         """Test that output of sampling method is an array when input data is a list"""
#         self.assertIsInstance(self.sampling_output_list, numpy.ndarray)

#     def test_output_dimentions(self):
#         """test that sampling method produces an array with the correct dimensions"""
#         self.assertEqual(self.sampling_output_dict.shape[0], SAMPLE_SIZE )
#         self.assertEqual(self.sampling_output_dict.shape[1], 3 )


if __name__ == '__main__':
    unittest.main()