"""
Created on 2020-09-22 16:07:04
Last modified on 2020-09-23 07:10:39

@author: L. F. Pereira (lfpereira@fe.up.pt))
"""

import pickle

import numpy as np

# imports
from abaqus import session  # NOQA

# # variable initialization and odb opening
# job_name = 'Simul_SUPERCOMPRESSIBLE_RIKS'
# odb_name = '{}.odb'.format(job_name)
# odb = session.openOdb(name=odb_name)


def main(odb):
    # variable initialization and odb opening
    # odb_name = '{}.odb'.format(job_name)
    # odb = session.openOdb(name=odb_name)

    riks_results = {}

    # reference point data
    variables = ["U", "UR", "RF", "RM"]
    set_name = "ZTOP_REF_POINT"
    step_name = "RIKS_STEP"
    step = odb.steps[step_name]
    directions = (1, 2, 3)
    nodes = odb.rootAssembly.nodeSets[set_name].nodes[0]
    # get variables
    for variable in variables:
        y = []
        for node in nodes:
            instance_name = (
                node.instanceName if node.instanceName else "ASSEMBLY"
            )
            name = "Node " + instance_name + "." + str(node.label)
            historyOutputs = step.historyRegions[name].historyOutputs
            node_data = []
            for direction in directions:
                node_data.append(
                    [
                        data[1]
                        for data in historyOutputs[
                            "%s%i" % (variable, direction)
                        ].data
                    ]
                )
            y.append(node_data)
        riks_results[variable] = np.array(y[0])

    # # deformation
    # frames = step.frames
    # nodeSet = odb.rootAssembly.elementSets[' ALL ELEMENTS']
    # directions = (1, 3,)
    # variable = 'E'
    # values = []
    # for frame in frames:
    #     varFieldOutputs = frame.fieldOutputs[variable]
    #     outputs = varFieldOutputs.getSubset(region=nodeSet).values
    #     output_frame = []
    #     for direction in directions:
    #         output_frame.append([output.data[direction - 1]
    #                             for output in outputs])
    #     values.append(output_frame)

    # riks_results[variable] = np.array(values)

    with open("results.pkl", "wb") as file:
        pickle.dump(riks_results, file)
