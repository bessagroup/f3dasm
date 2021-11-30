import unittest
import logging
import os
import numpy as np

from f3dasm.simulator.fenics_wrapper.problems.rve import RVE

class TestFenics(unittest.TestCase):
    """
    This test lets you confirm that if your McNemar Test's implementation correctly reject the hypothesis
    """
    def setUp(self) -> None:  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))

    def test_rve_solver(self):
        logging.info('Contents of rve test:')
        domain_filepath = os.path.join(self.cur_dir, 'resources/RVE2D/rve.xdmf')
        F11 = 1.    
        F12 = 1.
        F22 = 1.

        F_macro = np.array([[F11,F12],[F12,F22]])
        rve_problem = RVE({'solver': 'linear'}, domain_filename=domain_filepath, F_macro=F_macro, name='test')
        rve_problem.solve()
        # self.assertEqual()
    
if __name__ == '__main__':
    unittest.main()