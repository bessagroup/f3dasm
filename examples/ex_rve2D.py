import logging
import os
import uuid
import shutil
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.rve.rve_mesher import RVEMesher
from examples.rve.rve import RVE


cur_dir = os.path.dirname(os.path.realpath(__file__))

rve_mesher = RVEMesher(Lc=4, shape="Circle", size=[0.3])
uid = str(uuid.uuid1())
mesh_dir = os.path.join(cur_dir, uid)
rve_mesher.mesh(0.15, mesh_dir)

logging.info('Contents of rve test:')
domain_filepath = os.path.join(mesh_dir, 'rve.xdmf')
uid = str(uuid.uuid1())
work_dir = os.path.join(cur_dir, uid)

F11 = 1.    
F12 = 1.
F22 = 1.
dim = 2
F_macro = np.array([[F11,F12],[F12,F22]])
rve_problem = RVE({'solver': 'linear'}, domain_filename=domain_filepath, name='test')
rve_problem.solve(F_macro, work_dir)

S,_ = rve_problem.postprocess()

E = 0.5*(np.dot((F_macro+np.eye(dim)).T,(F_macro+np.eye(dim)))-np.eye(dim))
out = [E[0,0],E[0,1],E[1,1],S[0,0],S[0,1],S[1,1]]
print(out)

shutil.rmtree(mesh_dir)
shutil.rmtree(work_dir)