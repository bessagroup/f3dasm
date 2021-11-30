import unittest
import logging
import os
import uuid
import shutil
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from set_path import *

from f3dasm.simulator.fenics_wrapper.preprocessor.rve_mesher import RVEMesher
from f3dasm.simulator.fenics_wrapper.problems.rve import RVE2D

class TestFenics(unittest.TestCase):
    """
    This test lets you confirm that if your McNemar Test's implementation correctly reject the hypothesis
    """
    def setUp(self) -> None:  # This Method is executed once before each test
        logging.basicConfig(level=logging.DEBUG)
        self.cur_dir = os.path.dirname(os.path.realpath(__file__))


    def test_preprocessor(self):
        rve_mesher = RVEMesher(Lc=4, shape="Circle", size=[0.3])
        uid = str(uuid.uuid1())
        work_dir = os.path.join(self.cur_dir, uid)
        rve_mesher.mesh(0.15, work_dir)
        shutil.rmtree(work_dir)


    def test_rve_solver(self):
        logging.info('Contents of rve test:')
        domain_filepath = os.path.join(self.cur_dir, '../resources/RVE2D/rve.xdmf')
        uid = str(uuid.uuid1())
        work_dir = os.path.join(self.cur_dir, uid)

        F11 = 1.    
        F12 = 1.
        F22 = 1.
        dim = 2
        F_macro = np.array([[F11,F12],[F12,F22]])
        rve_problem = RVE2D({'solver': 'linear'}, domain_filename=domain_filepath, name='test')
        rve_problem.solve(F_macro, work_dir)
        
        S,_ = rve_problem.postprocess()

        E = 0.5*(np.dot((F_macro+np.eye(dim)).T,(F_macro+np.eye(dim)))-np.eye(dim))
        out = [E[0,0],E[0,1],E[1,1],S[0,0],S[0,1],S[1,1]]
        print(out)
        shutil.rmtree(work_dir)
        # self.assertEqual()
    
if __name__ == '__main__':
    unittest.main()