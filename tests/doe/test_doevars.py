##############################################################
## To test the F3DASM, install the path to the package      ##
## to the local python envionment as follows:               ##
## pip install -e .                                         ##
##############################################################

import unittest
from f3dasm.doe.doevars import DoeVars

RANGE_VARS = {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]} 
SAMPLE_SIZE = 5
IMPERFFECTIONS= 'imperfection values'

class TestDoeVars(unittest.TestCase):

    def setUp(self) -> None:
        """set up test fixtures"""
        self.doe_vars = DoeVars(SAMPLE_SIZE, RANGE_VARS, IMPERFFECTIONS)

    def test_post_init_fields(self):
        """Test that fields in the post_init are computed correctly"""
        self.assertListEqual(self.doe_vars.feature_names, ['F11', 'F12', 'F22'])
        self.assertEqual(self.doe_vars.dimensions, 3)
        self.assertEqual(self.doe_vars.values.shape, (SAMPLE_SIZE, 3) )

if __name__ == '__main__':
    unittest.main()