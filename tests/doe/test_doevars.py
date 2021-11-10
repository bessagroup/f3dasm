##############################################################
## To test the F3DASM, install the path to the package      ##
## to the local python envionment as follows:               ##
## pip install -e .                                         ##
##############################################################

import unittest
from f3dasm.doe.doevars import DoeVars
import pandas as pd

VARS = {
    'F11':[-0.15, 1], 
    'F12':[-0.1,0.15],
    'F22':[-0.15, 1], 
    'radius': [0.3, 5],  
    'material1': {'STEEL': {'E': [0,100], 'u': {0.1, 0.2, 0.3} }, 
                'CARBON': {'E': 5, 'u': 0.5, 's': 0.1 } },
    'material2': { 'CARBON': {'x': 2} },
    }

class TestRVE(unittest.TestCase):
    """Write relevant tests"""


class TestImperfections(unittest.TestCase):
    """Write relevant tests"""


class TestDoeVars(unittest.TestCase):

    def setUp(self) -> None:
        """set up test fixtures"""
        self.doe_vars = DoeVars(VARS)

    def test_as_dict(self):
        """Test that the as_dict method converts DoE variables in to a valid dictionary"""
        self.assertIsInstance(self.doe_vars.as_dict(), dict)

    def test_pandas_df(self):
        """Test that pandas_df method converts DoE variables into a valid pandas dataframe"""
        self.assertIsInstance(self.doe_vars.pandas_df(), pd.DataFrame)

if __name__ == '__main__':
    unittest.main()

