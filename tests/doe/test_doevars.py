##############################################################
## To test the F3DASM, install the path to the package      ##
## to the local python envionment as follows:               ##
## pip install -e .                                         ##
##############################################################

import unittest
from f3dasm.doe.doevars import DoeVars, Imperfection, Material, CircleMicrostructure, CilinderMicrostructure, RVE, BaseMicrosructure
import pandas as pd

B_CONDITIONS = {'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]} 
MICROSTRUCTURE_SIZE = 0.3
MATERIAL_A = {'parameter1': 0.5, 'paramter2': {0.1,0.2,0.3}}
MATERIAL_B = {'parameter1': 50, 'paramter2': 110}
SAMPLE_SIZE = 5
IMPERFFECTIONS= {'imp1': 1.2, 'imp2': {1,2,3}}
LC = 4
DIAMETER= 0.3
LENGTH = 1


class TestMaterial(unittest.TestCase):
    """Write relevant tests"""
    pass
    

class TestMicrostructures(unittest.TestCase):

    def setUp(self) -> None:
        self.micro_circle = CircleMicrostructure(MATERIAL_A, DIAMETER)
        self.micro_cilinder = CilinderMicrostructure(MATERIAL_A, DIAMETER, LENGTH)
        
    def test_circle_inheritance(self) -> None:
        """Test that  CircleMicrostruture inherits from BaseMicrostructure"""
        self.assertIsInstance(self.micro_circle, BaseMicrosructure)

    def test_cilinder_inheritance(self) -> None:
        """Test that  CilinderMicrostruture inherits from BaseMicrostructure"""
        self.assertIsInstance(self.micro_cilinder, BaseMicrosructure)

class TestRVE(unittest.TestCase):
    """Write relevant tests"""


class TestImperfections(unittest.TestCase):
    """Write relevant tests"""


class TestDoeVars(unittest.TestCase):

    def setUp(self) -> None:
        """set up test fixtures"""
        self.doe_vars = DoeVars(B_CONDITIONS, RVE(LC, Material(MATERIAL_A), CircleMicrostructure(MATERIAL_B, DIAMETER), dimesionality=2),Imperfection(IMPERFFECTIONS))

    def test_boundary_conditions(self):
        """Test that bournday contidions are named correctly"""
        self.assertListEqual(list(self.doe_vars.boundary_conditions.keys()), ['F11', 'F12', 'F22'])

    def test_conversion_to_dataframe(self):
        """Test if dataclass can be converted to a valid pandas dataframe"""
        pass

if __name__ == '__main__':
    unittest.main()

