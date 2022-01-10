import logging
import os
import uuid
import shutil
import numpy as np
import dolfin

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.rve.rve_mesher import RVEMesher
from examples.rve.rve import RVE

dolfin.list_linear_solver_methods()


cur_dir = os.path.dirname(os.path.realpath(__file__))

rve_mesher = RVEMesher(Lc=4, shape="Sphere", size=[0.3])
uid = str(uuid.uuid1())
mesh_dir = os.path.join(cur_dir, uid)
rve_mesher.mesh(0.3, mesh_dir)

logging.info('Contents of rve test:')
domain_filepath = os.path.join(mesh_dir, 'rve.xdmf')
uid = str(uuid.uuid1())
# uid = '0d921113-5a75-11ec-be75-60f262c28670'
work_dir = os.path.join(cur_dir, uid)

# doe_var = {'F11':[-0.1, 0.8], 'F12':[0.,0.15],'F13':[0., 0.15],\
#            'F21':[0., 0.15], 'F22':[-0.1,0.8],'F23':[0., 0.15],
#            'F31':[0., 0.15], 'F32':[0.,0.15],'F33':[-0.1, 0.8]}
F11 = 0.5   
F12 = 0.1
F13 = 0.1
F22 = 0.5
F23 = 0.1
F33 = 0.5

dim = 3
F_macro = np.array([[F11,F12,F13],[F12,F22,F23],[F13,F23,F33]])
rve_problem = RVE({'solver': 'linear'}, domain_filename=domain_filepath, name='test')
rve_problem.solve(F_macro, work_dir)

S,_ = rve_problem.postprocess()

E = 0.5*(np.dot((F_macro+np.eye(dim)).T,(F_macro+np.eye(dim)))-np.eye(dim))
out = [E[0,0],E[0,1],E[0,2],E[1,1],E[1,2],E[2,2],S[0,0],S[0,1],S[0,2],S[1,1],S[1,2],S[2,2]]
print(out)

shutil.rmtree(mesh_dir)
shutil.rmtree(work_dir)
