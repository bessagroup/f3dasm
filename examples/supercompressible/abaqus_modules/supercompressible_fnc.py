'''
Created on 2020-09-22 12:07:10
Last modified on 2020-09-30 07:41:13

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports

from __future__ import division

# abaqus
from caeModules import *  # allow noGui
from abaqus import mdb, backwardCompatibility
from abaqusConstants import (THREE_D, DEFORMABLE_BODY, IMPRINT, ON,
                             ANALYTIC_RIGID_SURFACE, ISOTROPIC, DURING_ANALYSIS,
                             LINEAR, MIDDLE_SURFACE, FROM_SECTION, N1_COSINES,
                             FINER, B31, CARTESIAN, WHOLE_SURFACE, OFF,
                             KINEMATIC, BUCKLING_MODES, HARD, FINITE)
from part import EdgeArray
import mesh
# from abaqus import session

# standard library
import itertools

# third-party
import numpy as np


# variable definition
# model_name = 'SUPERCOMPRESSIBLE_LIN_BUCKLE'
# job_name = 'Simul_SUPERCOMPRESSIBLE_LIN_BUCKLE'
# n_longerons = 3
# bottom_diameter = 100.
# top_diameter = 82.42
# pitch = 1.15223e2
# young_modulus = 3.5e3
# shear_modulus = 1.38631e3
# d = 10.
# imperfection = 7.85114e-02  # only needed for riks


# linear buckle

def lin_buckle(model_name, job_name, n_longerons, bottom_diameter,
               young_modulus, shear_modulus, ratio_top_diameter, ratio_pitch,
               ratio_d):

    # variables from ratios
    d = ratio_d * bottom_diameter
    pitch = ratio_pitch * bottom_diameter
    top_diameter = bottom_diameter * (1. - ratio_top_diameter)

    # variables with defaults
    n_storeys = 1
    twist_angle = 0.
    transition_length_ratio = 1.

    # compute variables
    mast_radius = bottom_diameter / 2.
    mast_height = n_storeys * pitch
    cone_slope = (bottom_diameter - top_diameter) / bottom_diameter

    # create abaqus model
    model = mdb.Model(name=model_name)
    backwardCompatibility.setValues(reportDeprecated=False)
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']

    # create joints
    joints = np.zeros((n_storeys + 1, n_longerons, 3))
    for i_storey in range(0, n_storeys + 1, 1):
        zcoord = mast_height / n_storeys * i_storey
        aux1 = 2.0 * np.pi / n_longerons
        aux2 = twist_angle * min(zcoord / mast_height / transition_length_ratio, 1.0)
        for i_vertex in range(0, n_longerons):
            aux3 = aux1 * i_vertex + aux2
            xcoord = mast_radius * np.cos(aux3)
            ycoord = mast_radius * np.sin(aux3)
            joints[i_storey, i_vertex, :] = (xcoord * (1.0 - min(zcoord, transition_length_ratio * mast_height) / mast_height * cone_slope), ycoord * (1.0 - min(zcoord, transition_length_ratio * mast_height) / mast_height * cone_slope), zcoord)

    # create geometry longerons
    longerons_name = 'LONGERONS'
    part_longerons = model.Part(longerons_name, dimensionality=THREE_D,
                                type=DEFORMABLE_BODY)
    longeron_points = []
    for i_vertex in range(0, n_longerons):
        # get required points
        longeron_points.append([joints[i_storey, i_vertex, :] for i_storey in range(0, n_storeys + 1)])
        # create wires
        part_longerons.WirePolyLine(points=longeron_points[-1],
                                    mergeType=IMPRINT, meshable=ON)

    # create surface
    surface_name = 'ANALYTICAL_SURF'
    s = model.ConstrainedSketch(name='SURFACE_SKETCH',
                                sheetSize=mast_radius * 3.0)
    s.Line(point1=(0.0, -mast_radius * 1.1),
           point2=(0.0, mast_radius * 1.1))
    part_surf = model.Part(name=surface_name, dimensionality=THREE_D,
                           type=ANALYTIC_RIGID_SURFACE)
    part_surf.AnalyticRigidSurfExtrude(sketch=s,
                                       depth=mast_radius * 2.2)

    # create required sets and surfaces
    # surface
    part_surf.Surface(side1Faces=part_surf.faces,
                      name=surface_name)

    # longeron
    edges = part_longerons.edges
    vertices = part_longerons.vertices

    # individual sets
    all_edges = []
    for i_vertex, long_pts in enumerate(longeron_points):
        # get vertices and edges
        selected_vertices = [vertices.findAt((pt,)) for pt in long_pts]
        all_edges.append(EdgeArray([edges.findAt(pt) for pt in long_pts]))
        # individual sets
        long_name = 'LONGERON-{}'.format(i_vertex)
        part_longerons.Set(edges=all_edges[-1], name=long_name)
        # joints
        for i_storey, vertex in enumerate(selected_vertices):
            joint_name = 'JOINT-{}-{}'.format(i_storey, i_vertex)
            part_longerons.Set(vertices=vertex, name=joint_name)

    name = 'ALL_LONGERONS'
    part_longerons.Set(edges=all_edges, name=name)
    name = 'ALL_LONGERONS_SURF'
    part_longerons.Surface(circumEdges=all_edges, name=name)

    # joint sets
    selected_vertices = []
    for i_storey in range(0, n_storeys + 1):
        selected_vertices.append([])
        for i_vertex in range(0, n_longerons):
            name = 'JOINT-{}-{}'.format(i_storey, i_vertex)
            selected_vertices[-1].append(part_longerons.sets[name].vertices)

    name = 'BOTTOM_JOINTS'
    part_longerons.Set(name=name, vertices=selected_vertices[0])
    name = 'TOP_JOINTS'
    part_longerons.Set(name=name, vertices=selected_vertices[-1])
    name = 'ALL_JOINTS'
    all_vertices = list(itertools.chain(*selected_vertices))
    part_longerons.Set(name=name, vertices=all_vertices)

    # create beam section
    # create section material
    material_name = 'LONGERON_MATERIAL'
    nu = young_modulus / (2 * shear_modulus) - 1
    abaqusMaterial = model.Material(name=material_name)
    abaqusMaterial.Elastic(type=ISOTROPIC, table=((young_modulus, nu),))
    # create profile
    profile_name = 'LONGERONS_PROFILE'
    r = d / 2.
    model.CircularProfile(name=profile_name, r=r)
    # create profile
    section_name = 'LONGERONS_SECTION'
    model.BeamSection(consistentMassMatrix=False, integration=DURING_ANALYSIS,
                      material=material_name, name=section_name,
                      poissonRatio=0.31, profile=profile_name,
                      temperatureVar=LINEAR)
    # section assignment
    part_longerons.SectionAssignment(
        offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
        region=part_longerons.sets['ALL_LONGERONS'],
        sectionName=section_name, thicknessAssignment=FROM_SECTION)
    # section orientation
    for i_vertex, pts in enumerate(longeron_points):
        dir_vec_n1 = np.array(pts[0]) - (0., 0., 0.)
        longeron_name = 'LONGERON-{}'.format(i_vertex)
        region = part_longerons.sets[longeron_name]
        part_longerons.assignBeamSectionOrientation(
            region=region, method=N1_COSINES, n1=dir_vec_n1)

    # generate mesh
    # seed part
    mesh_size = min(mast_radius, pitch) / 300.
    mesh_deviation_factor = .04
    mesh_min_size_factor = .001
    element_code = B31
    part_longerons.seedPart(
        size=mesh_size, deviationFactor=mesh_deviation_factor,
        minSizeFactor=mesh_min_size_factor, constraint=FINER)
    # assign element type
    elem_type_longerons = mesh.ElemType(elemCode=element_code)
    part_longerons.setElementType(regions=(part_longerons.edges,),
                                  elemTypes=(elem_type_longerons,))
    # generate mesh
    part_longerons.generateMesh()

    # create instances
    modelAssembly = model.rootAssembly
    part_surf = model.parts[surface_name]
    modelAssembly.Instance(name=longerons_name,
                           part=part_longerons, dependent=ON)
    modelAssembly.Instance(name=surface_name,
                           part=part_surf, dependent=ON)
    # rotate surface
    modelAssembly.rotate(instanceList=(surface_name, ),
                         axisPoint=(0., 0., 0.),
                         axisDirection=(0., 1., 0.), angle=90.)

    # create reference points for boundary conditions
    ref_point_positions = ['BOTTOM', 'TOP']
    for i, position in enumerate(ref_point_positions):
        sign = 1 if i else -1
        rp = modelAssembly.ReferencePoint(
            point=(0., 0., i * mast_height + sign * 1.1 * mast_radius))
        modelAssembly.Set(referencePoints=(modelAssembly.referencePoints[rp.id],),
                          name='Z{}_REF_POINT'.format(position))

    # add constraints for loading
    instance_longerons = modelAssembly.instances[longerons_name]
    instance_surf = modelAssembly.instances[surface_name]
    ref_points = [modelAssembly.sets['Z{}_REF_POINT'.format(position)]
                  for position in ref_point_positions]

    # bottom point and analytic surface
    surf = instance_surf.surfaces[surface_name]
    model.RigidBody('CONSTRAINT-RIGID_BODY-BOTTOM', refPointRegion=ref_points[0],
                    surfaceRegion=surf)

    # create local datums
    datums = []
    for i_vertex in range(0, n_longerons):
        origin = joints[0, i_vertex, :]
        point2 = joints[0, i_vertex - 1, :]
        name = 'LOCAL_DATUM_{}'.format(i_vertex)
        datums.append(part_longerons.DatumCsysByThreePoints(
            origin=origin, point2=point2, name=name, coordSysType=CARTESIAN,
            point1=(0.0, 0.0, 0.0)))

    # create coupling constraints
    for i_vertex in range(n_longerons):
        datum = instance_longerons.datums[datums[i_vertex].id]
        for i, i_storey in enumerate([0, n_storeys]):
            joint_name = 'JOINT-{}-{}'.format(i_storey, i_vertex)
            slave_region = instance_longerons.sets[joint_name]
            master_region = ref_points[i]
            constraint_name = 'CONSTRAINT-%s-%i-%i' % ('Z{}_REF_POINT'.format(ref_point_positions[i]),
                                                       i_storey, i_vertex)
            model.Coupling(name=constraint_name, controlPoint=master_region,
                           surface=slave_region, influenceRadius=WHOLE_SURFACE,
                           couplingType=KINEMATIC, localCsys=datum, u1=ON,
                           u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)

    # from now on, there's differences between linear buckle and riks

    # create step
    step_name = 'BUCKLE_STEP'
    model.BuckleStep(step_name, numEigen=20, previous='Initial', minEigen=0.)

    # set bcs (displacement)
    region_name = 'Z{}_REF_POINT'.format(ref_point_positions[0])
    loaded_region = modelAssembly.sets[region_name]
    model.DisplacementBC('BC_FIX', createStepName=step_name,
                         region=loaded_region, u1=0., u2=0., u3=0.,
                         ur1=0., ur2=0., ur3=0., buckleCase=BUCKLING_MODES)

    # set bcs (load)
    applied_load = -1.
    region_name = 'Z{}_REF_POINT'.format(ref_point_positions[-1])
    loaded_region = modelAssembly.sets[region_name]
    model.ConcentratedForce('APPLIED_FORCE', createStepName=step_name,
                            region=loaded_region, cf3=applied_load)

    # create provisory inp
    modelJob = mdb.Job(model=model_name, name=job_name)
    modelJob.writeInput(consistencyChecking=OFF)

    # ask for node file
    with open('{}.inp'.format(job_name), 'r') as file:
        lines = file.readlines()

    line_cmp = '** {}\n'.format('OUTPUT REQUESTS')
    for i, line in reversed(list(enumerate(lines))):
        if line == line_cmp:
            break

    insert_line = i + 2
    for line in reversed(['*NODE FILE, frequency=1', 'U']):
        lines.insert(insert_line, '{}\n'.format(line))

    with open('{}.inp'.format(job_name), 'w') as file:
        file.writelines(lines)

    # create job
    # modelJob = mdb.JobFromInputFile(inputFileName='{}.inp'.format(job_name),
    #                                 name=job_name)
    # modelJob.submit(consistencyChecking=OFF)
    # modelJob.waitForCompletion()


# linear buckle post-processing

# variable initialization and odb opening
# odb_name = '{}.odb'.format(job_name)
# odb = session.openOdb(name=odb_name)

def post_process_lin_buckle(odb):
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

    ur = ztop_ur[0]
    u = ztop_u[0]
    coilable = int(abs(ur[0][0]) > 1.0e-4 and abs(u[0][0]) < 1.0e-4 and abs(u[1][0]) < 1.0e-4)

    buckling_results = {'max_disps': max_disps,
                        'loads': eigenvalues,
                        'coilable': coilable}

    return buckling_results


# riks

# variable definition
# previous_model_job_name = job_name
# previous_model_results = buckling_results
# model_name = 'SUPERCOMPRESSIBLE_RIKS'
# job_name = 'Simul_SUPERCOMPRESSIBLE_RIKS'

# TODO: pass


def riks(model_name, job_name, n_longerons, bottom_diameter,
         young_modulus, shear_modulus, ratio_top_diameter, ratio_pitch,
         ratio_d, imperfection, previous_model_results, previous_model_job_name):

    # abort if not coilable
    if not int(previous_model_results['coilable']):
        return True

    # variables from ratios
    d = ratio_d * bottom_diameter
    pitch = ratio_pitch * bottom_diameter
    top_diameter = bottom_diameter * (1 - ratio_top_diameter)

    # variables with defaults
    n_storeys = 1
    twist_angle = 0.
    transition_length_ratio = 1.

    # compute variables
    mast_radius = bottom_diameter / 2.
    mast_height = n_storeys * pitch
    cone_slope = (bottom_diameter - top_diameter) / bottom_diameter

    # create abaqus model
    model = mdb.Model(name=model_name)
    backwardCompatibility.setValues(reportDeprecated=False)
    if 'Model-1' in mdb.models.keys():
        del mdb.models['Model-1']

    # create joints
    joints = np.zeros((n_storeys + 1, n_longerons, 3))
    for i_storey in range(0, n_storeys + 1, 1):
        zcoord = mast_height / n_storeys * i_storey
        aux1 = 2.0 * np.pi / n_longerons
        aux2 = twist_angle * min(zcoord / mast_height / transition_length_ratio, 1.0)
        for i_vertex in range(0, n_longerons):
            aux3 = aux1 * i_vertex + aux2
            xcoord = mast_radius * np.cos(aux3)
            ycoord = mast_radius * np.sin(aux3)
            joints[i_storey, i_vertex, :] = (xcoord * (1.0 - min(zcoord, transition_length_ratio * mast_height) / mast_height * cone_slope), ycoord * (1.0 - min(zcoord, transition_length_ratio * mast_height) / mast_height * cone_slope), zcoord)

    # create geometry longerons
    longerons_name = 'LONGERONS'
    part_longerons = model.Part(longerons_name, dimensionality=THREE_D,
                                type=DEFORMABLE_BODY)
    longeron_points = []
    for i_vertex in range(0, n_longerons):
        # get required points
        longeron_points.append([joints[i_storey, i_vertex, :] for i_storey in range(0, n_storeys + 1)])
        # create wires
        part_longerons.WirePolyLine(points=longeron_points[-1],
                                    mergeType=IMPRINT, meshable=ON)

    # create surface
    surface_name = 'ANALYTICAL_SURF'
    s = model.ConstrainedSketch(name='SURFACE_SKETCH',
                                sheetSize=mast_radius * 3.0)
    s.Line(point1=(0.0, -mast_radius * 1.1),
           point2=(0.0, mast_radius * 1.1))
    part_surf = model.Part(name=surface_name, dimensionality=THREE_D,
                           type=ANALYTIC_RIGID_SURFACE)
    part_surf.AnalyticRigidSurfExtrude(sketch=s,
                                       depth=mast_radius * 2.2)

    # create required sets and surfaces
    # surface
    part_surf.Surface(side1Faces=part_surf.faces,
                      name=surface_name)

    # longeron
    edges = part_longerons.edges
    vertices = part_longerons.vertices

    # individual sets
    all_edges = []
    for i_vertex, long_pts in enumerate(longeron_points):
        # get vertices and edges
        selected_vertices = [vertices.findAt((pt,)) for pt in long_pts]
        all_edges.append(EdgeArray([edges.findAt(pt) for pt in long_pts]))
        # individual sets
        long_name = 'LONGERON-{}'.format(i_vertex)
        part_longerons.Set(edges=all_edges[-1], name=long_name)
        # joints
        for i_storey, vertex in enumerate(selected_vertices):
            joint_name = 'JOINT-{}-{}'.format(i_storey, i_vertex)
            part_longerons.Set(vertices=vertex, name=joint_name)

    name = 'ALL_LONGERONS'
    part_longerons.Set(edges=all_edges, name=name)
    name = 'ALL_LONGERONS_SURF'
    part_longerons.Surface(circumEdges=all_edges, name=name)

    # joint sets
    selected_vertices = []
    for i_storey in range(0, n_storeys + 1):
        selected_vertices.append([])
        for i_vertex in range(0, n_longerons):
            name = 'JOINT-{}-{}'.format(i_storey, i_vertex)
            selected_vertices[-1].append(part_longerons.sets[name].vertices)

    name = 'BOTTOM_JOINTS'
    part_longerons.Set(name=name, vertices=selected_vertices[0])
    name = 'TOP_JOINTS'
    part_longerons.Set(name=name, vertices=selected_vertices[-1])
    name = 'ALL_JOINTS'
    all_vertices = list(itertools.chain(*selected_vertices))
    part_longerons.Set(name=name, vertices=all_vertices)

    # create beam section
    # create section material
    material_name = 'LONGERON_MATERIAL'
    nu = young_modulus / (2 * shear_modulus) - 1
    abaqusMaterial = model.Material(name=material_name)
    abaqusMaterial.Elastic(type=ISOTROPIC, table=((young_modulus, nu),))
    # create profile
    profile_name = 'LONGERONS_PROFILE'
    r = d / 2.
    model.CircularProfile(name=profile_name, r=r)
    # create profile
    section_name = 'LONGERONS_SECTION'
    model.BeamSection(consistentMassMatrix=False, integration=DURING_ANALYSIS,
                      material=material_name, name=section_name,
                      poissonRatio=0.31, profile=profile_name,
                      temperatureVar=LINEAR)
    # section assignment
    part_longerons.SectionAssignment(
        offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
        region=part_longerons.sets['ALL_LONGERONS'],
        sectionName=section_name, thicknessAssignment=FROM_SECTION)
    # section orientation
    for i_vertex, pts in enumerate(longeron_points):
        dir_vec_n1 = np.array(pts[0]) - (0., 0., 0.)
        longeron_name = 'LONGERON-{}'.format(i_vertex)
        region = part_longerons.sets[longeron_name]
        part_longerons.assignBeamSectionOrientation(
            region=region, method=N1_COSINES, n1=dir_vec_n1)

    # generate mesh
    # seed part
    mesh_size = min(mast_radius, pitch) / 300.
    mesh_deviation_factor = .04
    mesh_min_size_factor = .001
    element_code = B31
    part_longerons.seedPart(
        size=mesh_size, deviationFactor=mesh_deviation_factor,
        minSizeFactor=mesh_min_size_factor, constraint=FINER)
    # assign element type
    elem_type_longerons = mesh.ElemType(elemCode=element_code)
    part_longerons.setElementType(regions=(part_longerons.edges,),
                                  elemTypes=(elem_type_longerons,))
    # generate mesh
    part_longerons.generateMesh()

    # create instances
    modelAssembly = model.rootAssembly
    part_surf = model.parts[surface_name]
    modelAssembly.Instance(name=longerons_name,
                           part=part_longerons, dependent=ON)
    modelAssembly.Instance(name=surface_name,
                           part=part_surf, dependent=ON)
    # rotate surface
    modelAssembly.rotate(instanceList=(surface_name, ),
                         axisPoint=(0., 0., 0.),
                         axisDirection=(0., 1., 0.), angle=90.)

    # create reference points for boundary conditions
    ref_point_positions = ['BOTTOM', 'TOP']
    for i, position in enumerate(ref_point_positions):
        sign = 1 if i else -1
        rp = modelAssembly.ReferencePoint(
            point=(0., 0., i * mast_height + sign * 1.1 * mast_radius))
        modelAssembly.Set(referencePoints=(modelAssembly.referencePoints[rp.id],),
                          name='Z{}_REF_POINT'.format(position))

    # add constraints for loading
    instance_longerons = modelAssembly.instances[longerons_name]
    instance_surf = modelAssembly.instances[surface_name]
    ref_points = [modelAssembly.sets['Z{}_REF_POINT'.format(position)]
                  for position in ref_point_positions]

    # bottom point and analytic surface
    surf = instance_surf.surfaces[surface_name]
    model.RigidBody('CONSTRAINT-RIGID_BODY-BOTTOM', refPointRegion=ref_points[0],
                    surfaceRegion=surf)

    # create local datums
    datums = []
    for i_vertex in range(0, n_longerons):
        origin = joints[0, i_vertex, :]
        point2 = joints[0, i_vertex - 1, :]
        name = 'LOCAL_DATUM_{}'.format(i_vertex)
        datums.append(part_longerons.DatumCsysByThreePoints(
            origin=origin, point2=point2, name=name, coordSysType=CARTESIAN,
            point1=(0.0, 0.0, 0.0)))

    # create coupling constraints
    for i_vertex in range(n_longerons):
        datum = instance_longerons.datums[datums[i_vertex].id]
        for i, i_storey in enumerate([0, n_storeys]):
            joint_name = 'JOINT-{}-{}'.format(i_storey, i_vertex)
            slave_region = instance_longerons.sets[joint_name]
            master_region = ref_points[i]
            constraint_name = 'CONSTRAINT-%s-%i-%i' % ('Z{}_REF_POINT'.format(ref_point_positions[i]),
                                                       i_storey, i_vertex)
            model.Coupling(name=constraint_name, controlPoint=master_region,
                           surface=slave_region, influenceRadius=WHOLE_SURFACE,
                           couplingType=KINEMATIC, localCsys=datum, u1=ON,
                           u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)

    # from now on, there's differences between linear buckle and riks

    # create step
    step_name = 'RIKS_STEP'
    model.StaticRiksStep(step_name, nlgeom=ON, maxNumInc=400,
                         initialArcInc=5e-2, maxArcInc=0.5, previous='Initial')

    # set bcs (displacement) - shared with linear buckling
    region_name = 'Z{}_REF_POINT'.format(ref_point_positions[0])
    loaded_region = modelAssembly.sets[region_name]
    model.DisplacementBC('BC_FIX', createStepName=step_name,
                         region=loaded_region, u1=0., u2=0., u3=0.,
                         ur1=0., ur2=0., ur3=0., buckleCase=BUCKLING_MODES)

    # set bcs (displacement)
    vert_disp = - pitch
    region_name = 'Z{}_REF_POINT'.format(ref_point_positions[-1])
    loaded_region = modelAssembly.sets[region_name]
    model.DisplacementBC('DISPLACEMENT', createStepName=step_name,
                         region=loaded_region, u3=vert_disp,
                         buckleCase=BUCKLING_MODES)

    # set contact between longerons
    # add contact properties
    # contact property
    contact = model.ContactProperty('IMP_TARG')
    # contact behaviour
    contact.NormalBehavior(allowSeparation=OFF, pressureOverclosure=HARD)
    contact.GeometricProperties(contactArea=1., padThickness=None)
    # create interaction
    master = modelAssembly.instances[surface_name].surfaces[surface_name]
    slave = modelAssembly.instances[longerons_name].surfaces['ALL_LONGERONS_SURF']
    model.SurfaceToSurfaceContactStd(
        name='IMP_TARG', createStepName='Initial', master=master,
        slave=slave, sliding=FINITE, interactionProperty=contact.name,
        thickness=OFF)

    # outputs
    # energy outputs
    model.HistoryOutputRequest(
        name='ENERGIES', createStepName=step_name, variables=('ALLEN',))
    # load-disp outputs
    position = ref_point_positions[-1]
    region = model.rootAssembly.sets['Z{}_REF_POINT'.format(position)]
    model.HistoryOutputRequest(
        name='RP_{}'.format(position), createStepName=step_name,
        region=region, variables=('U', 'RF'))

    # create provisory inp
    modelJob = mdb.Job(model=model_name, name=job_name)
    modelJob.writeInput(consistencyChecking=OFF)

    # add imperfections to inp
    amp_factor = imperfection / previous_model_results['max_disps'][1]
    # TODO: deal with previous_model_job_name
    text = ['*IMPERFECTION, FILE={}, STEP=1'.format(previous_model_job_name),
            '{}, {}'.format(1, amp_factor)]
    with open('{}.inp'.format(job_name), 'r') as file:
        lines = file.readlines()

    line_cmp = '** {}\n'.format('INTERACTIONS')
    for i, line in reversed(list(enumerate(lines))):
        if line == line_cmp:
            break

    insert_line = i + 2
    for line in reversed(text):
        lines.insert(insert_line, '{}\n'.format(line))

    with open('{}.inp'.format(job_name), 'w') as file:
        file.writelines(lines)

    # create job
    # modelJob = mdb.JobFromInputFile(inputFileName='{}.inp'.format(job_name),
    #                                 name=job_name)
    # modelJob.submit(consistencyChecking=OFF)
    # modelJob.waitForCompletion()


# riks post-processing

# variable initialization and odb opening
# odb_name = '{}.odb'.format(job_name)
# odb = session.openOdb(name=odb_name)

def post_process_riks(odb):
    riks_results = {}

    # reference point data
    variables = ['U', 'UR', 'RF', 'RM']
    set_name = 'ZTOP_REF_POINT'
    step = odb.steps[odb.steps.keys()[-1]]
    directions = (1, 2, 3)
    nodes = odb.rootAssembly.nodeSets[set_name].nodes[0]
    # get variables
    for variable in variables:
        y = []
        for node in nodes:
            instance_name = node.instanceName if node.instanceName else 'ASSEMBLY'
            name = 'Node ' + instance_name + '.' + str(node.label)
            historyOutputs = step.historyRegions[name].historyOutputs
            node_data = []
            for direction in directions:
                node_data.append([data[1] for data in historyOutputs['%s%i' % (variable, direction)].data])
            y.append(node_data)
        riks_results[variable] = y[0]

    # deformation
    frames = step.frames
    nodeSet = odb.rootAssembly.elementSets[' ALL ELEMENTS']
    directions = (1, 3,)
    variable = 'E'
    values = []
    for frame in frames:
        varFieldOutputs = frame.fieldOutputs[variable]
        outputs = varFieldOutputs.getSubset(region=nodeSet).values
        output_frame = []
        for direction in directions:
            output_frame.append([output.data[direction - 1] for output in outputs])
        values.append(output_frame)

    riks_results[variable] = values

    return riks_results
