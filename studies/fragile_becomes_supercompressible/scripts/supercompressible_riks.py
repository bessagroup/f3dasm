# imports


# standard library
import itertools

import mesh

# third-party
import numpy as np
from abaqus import backwardCompatibility, mdb
from abaqusConstants import (
    ANALYTIC_RIGID_SURFACE,
    B31,
    BEFORE_ANALYSIS,
    BUCKLING_MODES,
    CARTESIAN,
    CONSTANT,
    DEFORMABLE_BODY,
    DURING_ANALYSIS,
    FINER,
    FINITE,
    FROM_SECTION,
    HARD,
    IMPRINT,
    ISOTROPIC,
    KINEMATIC,
    LINEAR,
    MIDDLE_SURFACE,
    N1_COSINES,
    NONE,
    OFF,
    OMIT,
    ON,
    THREE_D,
    WHOLE_SURFACE,
)

# abaqus
from caeModules import *  # allow noGui # NOQA
from part import EdgeArray


def main(dict):  # = 'lin_buckle'):
    model_name = "SUPERCOMPRESSIBLE_RIKS"
    job_number = "SUPERCOMPRESSIBLE_RIKS"  # int(dict['job_number'])

    # Flag
    circular = dict["circular"]  # True or False

    # Constant parameters DoE
    n_longerons = int(dict["n_longerons"])
    young_modulus = dict["young_modulus"]
    bottom_diameter = dict["bottom_diameter"]

    # Variables from DoE
    ratio_pitch = dict["ratio_pitch"]
    ratio_top_diameter = dict["ratio_top_diameter"]
    ratio_shear_modulus = dict["ratio_shear_modulus"]

    # Output from linear buckling analysis
    # coilable = dict['coilable']
    lin_bckl_max_disp = dict["max_disps"]
    lin_buckle_odb = dict["lin_buckle_odb"]

    # Imperfection
    imperfection = dict["imperfection"]

    if circular:
        ratio_d = dict["ratio_d"]
        d = ratio_d * bottom_diameter

    # 7D
    if not circular:
        ratio_area = dict["ratio_area"]
        ratio_Ixx = dict["ratio_Ixx"]
        ratio_Iyy = dict["ratio_Iyy"]
        ratio_J = dict["ratio_J"]
        # initialization
        area = ratio_area * bottom_diameter**2
        Ixx = ratio_Ixx * bottom_diameter**4
        Iyy = ratio_Iyy * bottom_diameter**4
        J = ratio_J * bottom_diameter**4

    # variables from ratios
    shear_modulus = ratio_shear_modulus * young_modulus
    pitch = ratio_pitch * bottom_diameter
    top_diameter = bottom_diameter * (1 - ratio_top_diameter)

    # variables with defaults
    n_storeys = 1
    twist_angle = 0.0
    transition_length_ratio = 1.0

    # compute variables
    mast_radius = bottom_diameter / 2.0
    mast_height = n_storeys * pitch
    cone_slope = (bottom_diameter - top_diameter) / bottom_diameter

    # create abaqus model
    model = mdb.Model(name=model_name)
    backwardCompatibility.setValues(reportDeprecated=False)
    if "Model-1" in mdb.models.keys():
        del mdb.models["Model-1"]

    # create joints
    joints = np.zeros((n_storeys + 1, n_longerons, 3))
    for i_storey in range(0, n_storeys + 1, 1):
        zcoord = mast_height / n_storeys * i_storey
        aux1 = 2.0 * np.pi / n_longerons
        aux2 = twist_angle * min(
            zcoord / mast_height / transition_length_ratio, 1.0
        )
        for i_vertex in range(0, n_longerons):
            aux3 = aux1 * i_vertex + aux2
            xcoord = mast_radius * np.cos(aux3)
            ycoord = mast_radius * np.sin(aux3)
            joints[i_storey, i_vertex, :] = (
                xcoord
                * (
                    1.0
                    - min(zcoord, transition_length_ratio * mast_height)
                    / mast_height
                    * cone_slope
                ),
                ycoord
                * (
                    1.0
                    - min(zcoord, transition_length_ratio * mast_height)
                    / mast_height
                    * cone_slope
                ),
                zcoord,
            )

    # create geometry longerons
    longerons_name = "LONGERONS"
    part_longerons = model.Part(
        longerons_name, dimensionality=THREE_D, type=DEFORMABLE_BODY
    )
    longeron_points = []
    for i_vertex in range(0, n_longerons):
        # get required points
        longeron_points.append(
            [
                joints[i_storey, i_vertex, :]
                for i_storey in range(0, n_storeys + 1)
            ]
        )
        # create wires
        part_longerons.WirePolyLine(
            points=longeron_points[-1], mergeType=IMPRINT, meshable=ON
        )

    # create surface
    surface_name = "ANALYTICAL_SURF"
    s = model.ConstrainedSketch(
        name="SURFACE_SKETCH", sheetSize=mast_radius * 3.0
    )
    s.Line(point1=(0.0, -mast_radius * 1.1), point2=(0.0, mast_radius * 1.1))
    part_surf = model.Part(
        name=surface_name, dimensionality=THREE_D, type=ANALYTIC_RIGID_SURFACE
    )
    part_surf.AnalyticRigidSurfExtrude(sketch=s, depth=mast_radius * 2.2)

    # create required sets and surfaces
    # surface
    part_surf.Surface(side1Faces=part_surf.faces, name=surface_name)

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
        long_name = f"LONGERON-{i_vertex}"
        part_longerons.Set(edges=all_edges[-1], name=long_name)
        # joints
        for i_storey, vertex in enumerate(selected_vertices):
            joint_name = f"JOINT-{i_storey}-{i_vertex}"
            part_longerons.Set(vertices=vertex, name=joint_name)

    name = "ALL_LONGERONS"
    part_longerons.Set(edges=all_edges, name=name)
    name = "ALL_LONGERONS_SURF"
    part_longerons.Surface(circumEdges=all_edges, name=name)

    # joint sets
    selected_vertices = []
    for i_storey in range(0, n_storeys + 1):
        selected_vertices.append([])
        for i_vertex in range(0, n_longerons):
            name = f"JOINT-{i_storey}-{i_vertex}"
            selected_vertices[-1].append(part_longerons.sets[name].vertices)

    name = "BOTTOM_JOINTS"
    part_longerons.Set(name=name, vertices=selected_vertices[0])
    name = "TOP_JOINTS"
    part_longerons.Set(name=name, vertices=selected_vertices[-1])
    name = "ALL_JOINTS"
    all_vertices = list(itertools.chain(*selected_vertices))
    part_longerons.Set(name=name, vertices=all_vertices)

    # create beam section
    # create section material
    material_name = "LONGERON_MATERIAL"
    nu = young_modulus / (2 * shear_modulus) - 1
    abaqusMaterial = model.Material(name=material_name)
    abaqusMaterial.Elastic(type=ISOTROPIC, table=((young_modulus, nu),))

    # create profile
    profile_name = "LONGERONS_PROFILE"
    section_name = "LONGERONS_SECTION"

    if circular:
        # 3D simplified model
        r = d / 2.0
        model.CircularProfile(name=profile_name, r=r)
        # create profile
        model.BeamSection(
            consistentMassMatrix=False,
            integration=DURING_ANALYSIS,
            material=material_name,
            name=section_name,
            poissonRatio=0.31,
            profile=profile_name,
            temperatureVar=LINEAR,
        )

    if not circular:
        # 7D full model
        # create profile
        model.GeneralizedProfile(
            name=profile_name,
            area=area,
            i11=Ixx,
            i12=0.0,
            i22=Iyy,
            j=J,
            gammaO=0.0,
            gammaW=0.0,
        )
        # create section
        model.BeamSection(
            name=section_name,
            integration=BEFORE_ANALYSIS,
            beamShape=CONSTANT,
            profile=profile_name,
            thermalExpansion=OFF,
            temperatureDependency=OFF,
            dependencies=0,
            table=((young_modulus, shear_modulus),),
            poissonRatio=0.31,
            alphaDamping=0.0,
            betaDamping=0.0,
            compositeDamping=0.0,
            centroid=(0.0, 0.0),
            shearCenter=(0.0, 0.0),
            consistentMassMatrix=False,
        )

    # section assignment
    part_longerons.SectionAssignment(
        offset=0.0,
        offsetField="",
        offsetType=MIDDLE_SURFACE,
        region=part_longerons.sets["ALL_LONGERONS"],
        sectionName=section_name,
        thicknessAssignment=FROM_SECTION,
    )
    # section orientation
    for i_vertex, pts in enumerate(longeron_points):
        dir_vec_n1 = np.array(pts[0]) - (0.0, 0.0, 0.0)
        longeron_name = f"LONGERON-{i_vertex}"
        region = part_longerons.sets[longeron_name]
        part_longerons.assignBeamSectionOrientation(
            region=region, method=N1_COSINES, n1=dir_vec_n1
        )

    # generate mesh
    # seed part
    mesh_size = min(mast_radius, pitch) / 300.0
    mesh_deviation_factor = 0.04
    mesh_min_size_factor = 0.001
    element_code = B31
    part_longerons.seedPart(
        size=mesh_size,
        deviationFactor=mesh_deviation_factor,
        minSizeFactor=mesh_min_size_factor,
        constraint=FINER,
    )
    # assign element type
    elem_type_longerons = mesh.ElemType(elemCode=element_code)
    part_longerons.setElementType(
        regions=(part_longerons.edges,), elemTypes=(elem_type_longerons,)
    )
    # generate mesh
    part_longerons.generateMesh()

    # create instances
    modelAssembly = model.rootAssembly
    part_surf = model.parts[surface_name]
    modelAssembly.Instance(
        name=longerons_name, part=part_longerons, dependent=ON
    )
    modelAssembly.Instance(name=surface_name, part=part_surf, dependent=ON)
    # rotate surface
    modelAssembly.rotate(
        instanceList=(surface_name,),
        axisPoint=(0.0, 0.0, 0.0),
        axisDirection=(0.0, 1.0, 0.0),
        angle=90.0,
    )

    # create reference points for boundary conditions
    ref_point_positions = ["BOTTOM", "TOP"]
    for i, position in enumerate(ref_point_positions):
        sign = 1 if i else -1
        rp = modelAssembly.ReferencePoint(
            point=(0.0, 0.0, i * mast_height + sign * 1.1 * mast_radius)
        )
        modelAssembly.Set(
            referencePoints=(modelAssembly.referencePoints[rp.id],),
            name=f"Z{position}_REF_POINT",
        )

    # add constraints for loading
    instance_longerons = modelAssembly.instances[longerons_name]
    instance_surf = modelAssembly.instances[surface_name]
    ref_points = [
        modelAssembly.sets[f"Z{position}_REF_POINT"]
        for position in ref_point_positions
    ]

    # bottom point and analytic surface
    surf = instance_surf.surfaces[surface_name]
    model.RigidBody(
        "CONSTRAINT-RIGID_BODY-BOTTOM",
        refPointRegion=ref_points[0],
        surfaceRegion=surf,
    )

    # create local datums
    datums = []
    for i_vertex in range(0, n_longerons):
        origin = joints[0, i_vertex, :]
        point2 = joints[0, i_vertex - 1, :]
        name = f"LOCAL_DATUM_{i_vertex}"
        datums.append(
            part_longerons.DatumCsysByThreePoints(
                origin=origin,
                point2=point2,
                name=name,
                coordSysType=CARTESIAN,
                point1=(0.0, 0.0, 0.0),
            )
        )

    # create coupling constraints
    for i_vertex in range(n_longerons):
        datum = instance_longerons.datums[datums[i_vertex].id]
        for i, i_storey in enumerate([0, n_storeys]):
            joint_name = f"JOINT-{i_storey}-{i_vertex}"
            slave_region = instance_longerons.sets[joint_name]
            master_region = ref_points[i]
            constraint_name = "CONSTRAINT-%s-%i-%i" % (
                f"Z{ref_point_positions[i]}_REF_POINT",
                i_storey,
                i_vertex,
            )
            model.Coupling(
                name=constraint_name,
                controlPoint=master_region,
                surface=slave_region,
                influenceRadius=WHOLE_SURFACE,
                couplingType=KINEMATIC,
                localCsys=datum,
                u1=ON,
                u2=ON,
                u3=ON,
                ur1=OFF,
                ur2=ON,
                ur3=ON,
            )

    # from now on, there's differences between linear buckle and riks

    # create step
    step_name = "RIKS_STEP"
    model.StaticRiksStep(
        step_name,
        nlgeom=ON,
        maxNumInc=400,
        initialArcInc=5e-2,
        maxArcInc=0.5,
        previous="Initial",
    )

    # set bcs (displacement) - shared with linear buckling
    region_name = f"Z{ref_point_positions[0]}_REF_POINT"
    loaded_region = modelAssembly.sets[region_name]
    model.DisplacementBC(
        "BC_FIX",
        createStepName=step_name,
        region=loaded_region,
        u1=0.0,
        u2=0.0,
        u3=0.0,
        ur1=0.0,
        ur2=0.0,
        ur3=0.0,
        buckleCase=BUCKLING_MODES,
    )

    # set bcs (displacement)
    vert_disp = -pitch
    region_name = f"Z{ref_point_positions[-1]}_REF_POINT"
    loaded_region = modelAssembly.sets[region_name]
    model.DisplacementBC(
        "DISPLACEMENT",
        createStepName=step_name,
        region=loaded_region,
        u3=vert_disp,
        buckleCase=BUCKLING_MODES,
    )

    # set contact between longerons
    # add contact properties
    # contact property
    contact = model.ContactProperty("IMP_TARG")
    # contact behaviour
    contact.NormalBehavior(allowSeparation=OFF, pressureOverclosure=HARD)
    contact.GeometricProperties(contactArea=1.0, padThickness=None)
    # create interaction
    master = modelAssembly.instances[surface_name].surfaces[surface_name]
    slave = modelAssembly.instances[longerons_name].surfaces[
        "ALL_LONGERONS_SURF"
    ]
    # model.SurfaceToSurfaceContactStd(name='IMP_TARG',
    # createStepName='Initial', master=master,
    #                                  slave=slave, sliding=FINITE,
    # interactionProperty=contact.name, thickness=OFF)

    # model.SurfaceToSurfaceContactStd(
    #    name='IMP_TARG', createStepName='Initial', master=master,
    #    slave=slave, sliding=FINITE, interactionProperty=contact.name,
    #    thickness=OFF)
    model.SurfaceToSurfaceContactStd(
        name="IMP_TARG",
        adjustMethod=NONE,
        clearanceRegion=None,
        createStepName="Initial",
        datumAxis=None,
        initialClearance=OMIT,
        interactionProperty=contact.name,
        main=master,
        secondary=slave,
        sliding=FINITE,
        thickness=OFF,
    )

    # outputs
    # energy outputs
    model.HistoryOutputRequest(
        name="ENERGIES", createStepName=step_name, variables=("ALLEN",)
    )
    # load-disp outputs
    position = ref_point_positions[-1]
    region = model.rootAssembly.sets[f"Z{position}_REF_POINT"]
    model.HistoryOutputRequest(
        name=f"RP_{position}",
        createStepName=step_name,
        region=region,
        variables=("U", "RF"),
    )

    # create provisory inp
    modelJob = mdb.Job(model=model_name, name=str(job_number))
    modelJob.writeInput(consistencyChecking=OFF)

    # add imperfections to inp
    # previous_model_results['max_disps'][1]
    amp_factor = imperfection / lin_bckl_max_disp[1]
    # TODO: deal with previous_model_job_name
    text = [
        f"*IMPERFECTION, FILE={lin_buckle_odb}, STEP=1",
        f"{1}, {amp_factor}",
    ]
    with open(f"{job_number}.inp") as file:
        lines = file.readlines()

    line_cmp = "** {}\n".format("INTERACTIONS")
    for i, line in reversed(list(enumerate(lines))):
        if line == line_cmp:
            break

    insert_line = i + 2
    for line in reversed(text):
        lines.insert(insert_line, f"{line}\n")

    with open(f"{job_number}.inp", "w") as file:
        file.writelines(lines)

    # # create job
    # modelJob = mdb.JobFromInputFile(inputFileName='{}.inp'.format(job_name),
    #                                 name=job_name)
    # modelJob.submit(consistencyChecking=OFF)
    # modelJob.waitForCompletion()
