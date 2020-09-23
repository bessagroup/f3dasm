'''
Created on 2020-09-22 15:35:10
Last modified on 2020-09-23 07:11:00

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports
from abaqus import session


# variable initialization and odb opening
job_name = 'Simul_SUPERCOMPRESSIBLE_LIN_BUCKLE'
odb_name = '{}.odb'.format(job_name)
odb = session.openOdb(name=odb_name)

# initialization
step = odb.steps[odb.steps.keys()[-1]]
frames = step.frames

# get maximum displacements
variable = 'UR'
directions = (1, 2, 3)
nodeSet = odb.rootAssembly.nodeSets[' ALL NODES']
values = []
for frame in frames:
    varFieldOutputs = frame.fieldOutputs[variable]
    outputs = varFieldOutputs.getSubset(region=nodeSet).values
    output_frame = []
    for direction in directions:
        output_frame.append([output.data[direction - 1] for output in outputs])
    values.append(output_frame)

max_disps = []
for value in values:
    max_disp = np.max(np.abs(np.array(value)))
    max_disps.append(max_disp)


# get loads
eigenvalues = [float(frame.description.split('EigenValue =')[1]) for frame in list(frames)[1:]]

# is coilable
# get top ref point info
ztop_set_name = 'ZTOP_REF_POINT'
nodeSet = odb.rootAssembly.nodeSets[ztop_set_name]

# get info
directions = (3,)
variable = 'UR'
ztop_ur = []
for frame in list(frames)[1:]:
    varFieldOutputs = frame.fieldOutputs[variable]
    outputs = varFieldOutputs.getSubset(region=nodeSet).values
    output_frame = []
    for direction in directions:
        output_frame.append([output.data[direction - 1] for output in outputs])
    ztop_ur.append(output_frame)

directions = (1, 2,)
variable = 'U'
ztop_u = []
for frame in list(frames)[1:]:
    varFieldOutputs = frame.fieldOutputs[variable]
    outputs = varFieldOutputs.getSubset(region=nodeSet).values
    output_frame = []
    for direction in directions:
        output_frame.append([output.data[direction - 1] for output in outputs])
    ztop_u.append(output_frame)

coilable = [int(abs(ur[0][0]) > 1.0e-4 and abs(u[0][0]) < 1.0e-4 and abs(u[1][0]) < 1.0e-4)
            for ur, u in zip(ztop_ur, ztop_u)]

buckling_results = {'max_disps': max_disps,
                    'loads': eigenvalues,
                    'coilable': coilable}
