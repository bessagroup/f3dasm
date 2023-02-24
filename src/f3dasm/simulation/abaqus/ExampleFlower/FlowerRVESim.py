import json
from math import *

import numpy as np
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup


def main(sim_info):
    """

    Parameters
    ----------
    input_info:
    input_info= {'C1': float,
                'C2': float,
                'MAT_Name': string
                'job_name': str
                }

    Returns
    -------
    """
    # define the design variables for the shape of the flower
    C1 = float(sim_info["C1"])
    C2 = float(sim_info["C2"])
    job_name = str(sim_info["job_name"])
    mat_name = str(sim_info["MAT_Name"])
    # C1 = 1.99419533e-01
    # C2 = 7.24177347e-02
    # The fixed geometry parameters of the rve
    Lx = 3.50000  # sample dimention in um VF9 and 28%
    Ly = 3.50000  # sample dimention in um VF9 and 28%
    RVEcenter = [1.75000, 1.75000]  # Center position of RVE
    Mesh_size = 0.05000  # Mesh parameter
    ### MODEL PARAMETERS ###
    # NUMBER OF POINTS TO GENERATE THE CENTER HOLE THROUGH THE PARAMETRIC FUNCTION
    NUMBER_OF_POINTS = 100
    RO = 1.0
    TOL = 1e-5

    # define names of modeling

    # Begin to construct the model
    executeOnCaeStartup()
    session.viewports["Viewport: 1"].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)
    Mdb()
    model = mdb.models["Model-1"]

    sketch = model.ConstrainedSketch(
        name="__profile__",
        sheetSize=10.50000,
    )
    # g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    sketch.setPrimaryObject(option=STANDALONE)
    sketch.rectangle(
        point1=(RVEcenter[0] - Lx / 2, RVEcenter[1] - Ly / 2), point2=(RVEcenter[0] + Lx / 2, RVEcenter[1] + Ly / 2)
    )

    ### GENERATING INNER SHAPE OF THE GEOMETRY ###
    THETAALL = np.linspace(0.0, 2.0 * pi, NUMBER_OF_POINTS)
    POINTS = []
    for i in xrange(NUMBER_OF_POINTS):
        THETA = THETAALL[i]
        rr = RO * (1.0 + C1 * cos(4.0 * THETA) + C2 * cos(8.0 * THETA))
        POINTS.append((RVEcenter[0] + rr * cos(THETA),
                      RVEcenter[1] + rr * sin(THETA)))
        if i == 0:
            xFirst = RVEcenter[0] + rr * cos(THETA)
            yFirst = RVEcenter[1] + rr * sin(THETA)
        if i == NUMBER_OF_POINTS - 1:
            POINTS.append((xFirst, yFirst))

    ### GENERATE SPLINE SHAPE ###
    sketch.Spline(points=POINTS)
    part = model.Part(
        name="FinalRVE", dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del model.sketches["__profile__"]

    ### CREATING ASSEMBLY ###
    model_assembly = model.rootAssembly
    # INSTANCE DEFINITION
    Instance_Full = model_assembly.Instance(
        dependent=ON, name="FinalRVE", part=part)

    # create sets for faces, edges, and vertexes

    # All faces:
    f = part.faces
    faces = f[:]
    part.Set(faces=faces, name="Phase_1")

    # Create sets useful for meshing
    delta = min((min(Lx, Ly) / 1000), Mesh_size / 10)
    # edges
    s = part.edges
    edgesLEFT = s.getByBoundingBox(
        RVEcenter[0] - Lx / 2 - delta,
        RVEcenter[1] - Ly / 2 - delta,
        0,
        RVEcenter[0] - Lx / 2 + delta,
        RVEcenter[1] + Ly / 2 + delta,
        0,
    )
    part.Set(edges=edgesLEFT, name="LeftEdge")
    edgesRIGHT = s.getByBoundingBox(
        RVEcenter[0] + Lx / 2 - delta,
        RVEcenter[1] - Ly / 2 - delta,
        0,
        RVEcenter[0] + Lx / 2 + delta,
        RVEcenter[1] + Ly / 2 + delta,
        0,
    )
    part.Set(edges=edgesRIGHT, name="RightEdge")
    edgesTOP = s.getByBoundingBox(
        RVEcenter[0] - Lx / 2 - delta,
        RVEcenter[1] + Ly / 2 - delta,
        0,
        RVEcenter[0] + Lx / 2 + delta,
        RVEcenter[1] + Ly / 2 + delta,
        0,
    )
    part.Set(edges=edgesTOP, name="TopEdge")
    edgesBOT = s.getByBoundingBox(
        RVEcenter[0] - Lx / 2 - delta,
        RVEcenter[1] - Ly / 2 - delta,
        0,
        RVEcenter[0] + Lx / 2 + delta,
        RVEcenter[1] - Ly / 2 + delta,
        0,
    )
    part.Set(edges=edgesBOT, name="BotEdge")
    # vertices
    v = part.vertices
    vertexLB = v.getByBoundingBox(
        RVEcenter[0] - Lx / 2 - delta,
        RVEcenter[1] - Ly / 2 - delta,
        0,
        RVEcenter[0] - Lx / 2 + delta,
        RVEcenter[1] - Ly / 2 + delta,
        0,
    )
    part.Set(vertices=vertexLB, name="VertexLB")
    vertexRB = v.getByBoundingBox(
        RVEcenter[0] + Lx / 2 - delta,
        RVEcenter[1] - Ly / 2 - delta,
        0,
        RVEcenter[0] + Lx / 2 + delta,
        RVEcenter[1] - Ly / 2 + delta,
        0,
    )
    part.Set(vertices=vertexRB, name="VertexRB")
    vertexRT = v.getByBoundingBox(
        RVEcenter[0] + Lx / 2 - delta,
        RVEcenter[1] + Ly / 2 - delta,
        0,
        RVEcenter[0] + Lx / 2 + delta,
        RVEcenter[1] + Ly / 2 + delta,
        0,
    )
    part.Set(vertices=vertexRT, name="VertexRT")
    vertexLT = v.getByBoundingBox(
        RVEcenter[0] - Lx / 2 - delta,
        RVEcenter[1] + Ly / 2 - delta,
        0,
        RVEcenter[0] - Lx / 2 + delta,
        RVEcenter[1] + Ly / 2 + delta,
        0,
    )
    part.Set(vertices=vertexLT, name="VertexLT")

    ##########################################################
    # create materials

    if mat_name == "Arruda":
        material = model.Material(name="Arruda")
        material.Density(table=((1e-21,),))
        material.Expansion(table=((5.8e-05, 0.0), (5.8e-05, 200.0)),
                           zero=120.0, temperatureDependency=ON)
        material.Hyperelastic(
            materialType=ISOTROPIC,
            testData=OFF,
            type=ARRUDA_BOYCE,
            volumetricResponse=VOLUMETRIC_DATA,
            table=((166.0, 2.8, 0.0025),),
        )
        model.HomogeneousSolidSection(
            name="Section-1", material="Arruda", thickness=None)
        part.SectionAssignment(
            region=part.sets["Phase_1"],
            sectionName="Section-1",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )

    elif mat_name == "Neohookean":
        material = model.Material(name="neohookean")
        material.Density(table=((1e-21,),))
        material.Expansion(table=((5.8e-05, 0.0), (5.8e-05, 200.0)),
                           zero=120.0, temperatureDependency=ON)
        material.Hyperelastic(
            materialType=ISOTROPIC,
            testData=OFF,
            type=NEO_HOOKE,
            volumetricResponse=VOLUMETRIC_DATA,
            table=((961.538, 0.0005),),
        )
        model.HomogeneousSolidSection(
            name="Section-1", material="neohookean", thickness=None)
        part.SectionAssignment(
            region=part.sets["Phase_1"],
            sectionName="Section-1",
            offset=0.0,
            offsetType=MIDDLE_SURFACE,
            offsetField="",
            thicknessAssignment=FROM_SECTION,
        )
    else:
        print("Name of the material is wrong !")

    # define the dummy node by using the reference points
    RF_Right_id = model_assembly.ReferencePoint(
        point=(RVEcenter[0] + Lx / 2, 0.0, 0.0)).id
    RF_Top_id = model_assembly.ReferencePoint(
        point=(0.0, RVEcenter[1] + Ly / 2, 0.0)).id
    refpoints = model_assembly.referencePoints
    model_assembly.Set(
        name="Ref-R", referencePoints=((refpoints[RF_Right_id],)))
    model_assembly.Set(name="Ref-T", referencePoints=((refpoints[RF_Top_id],)))

    # mesh for the RVE
    niter = 1  # iteration number for meshing procedure
    status_mesh = 0  # flag signaling if mesh was created
    # Parameter used to refine mesh (should be larger than 1)
    refine_factor = 1.25000

    def get_node_y(node):
        return node.coordinates[1]

    def get_node_x(node):
        return node.coordinates[0]

    while status_mesh == 0:
        import mesh

        elemType1 = mesh.ElemType(
            elemCode=CPS4R,
            elemLibrary=STANDARD,
            secondOrderAccuracy=OFF,
            hourglassControl=ENHANCED,
            distortionControl=DEFAULT,
        )
        elemType2 = mesh.ElemType(elemCode=CPS3, elemLibrary=STANDARD)
        part.setElementType(regions=(faces,), elemTypes=(elemType1, elemType2))
        part.seedPart(size=Mesh_size, deviationFactor=0.4, minSizeFactor=0.4)
        part.seedEdgeBySize(edges=edgesLEFT, size=Mesh_size,
                            deviationFactor=0.4, constraint=FIXED)
        part.seedEdgeBySize(edges=edgesRIGHT, size=Mesh_size,
                            deviationFactor=0.4, constraint=FIXED)
        part.seedEdgeBySize(edges=edgesTOP, size=Mesh_size,
                            deviationFactor=0.4, constraint=FIXED)
        part.seedEdgeBySize(edges=edgesBOT, size=Mesh_size,
                            deviationFactor=0.4, constraint=FIXED)
        part.generateMesh()

        # judge if the nodes of each edge pair are the same or not
        BotEdge_nodes = part.sets["BotEdge"].nodes
        BotEdge_nodes_sorted = sorted(BotEdge_nodes, key=get_node_x)
        TopEdge_nodes = part.sets["TopEdge"].nodes
        TopEdge_nodes_sorted = sorted(TopEdge_nodes, key=get_node_x)

        LeftEdge_nodes = part.sets["LeftEdge"].nodes
        LeftEdge_nodes_sorted = sorted(LeftEdge_nodes, key=get_node_y)
        RightEdge_nodes = part.sets["RightEdge"].nodes
        RightEdge_nodes_sorted = sorted(RightEdge_nodes, key=get_node_y)
        #
        # Check if the node count in the Bottom and Top edges is the same:
        if len(BotEdge_nodes_sorted) != len(TopEdge_nodes_sorted):
            status_mesh = 0
            # If we have too many iterations then print error to ERROR file
            if niter <= 3:
                niter = niter + 1
                Mesh_size = Mesh_size / refine_factor  # refine mesh
            else:
                status_mesh = 2  # Did not find valid mesh...
                ERROR_file = open("ERROR_FILE", "a")
                # writing the entered content to the end of the ERROR_FILE
                ERROR_file.write("Failed to mesh the RVE for this sample\n")
                ERROR_file.close()
        elif len(LeftEdge_nodes_sorted) != len(RightEdge_nodes_sorted):
            status_mesh = 0
            # If we have too many iterations then print error to ERROR file
            if niter <= 3:
                niter = niter + 1
                Mesh_size = Mesh_size / refine_factor  # refine mesh
            else:
                status_mesh = 2  # Did not find valid mesh...
                ERROR_file = open("ERROR_FILE", "a")
                # writing the entered content to the end of the ERROR_FILE
                ERROR_file.write("Failed to mesh the RVE for this sample\n")
                ERROR_file.close()
        else:
            status_mesh = 1

    # create PBC for RVE
    import assembly

    session.viewports["Viewport: 1"].setValues(displayedObject=model_assembly)
    model_assembly.regenerate()
    # find out the Vertices
    NodeLB = part.sets["VertexLB"].nodes
    model_assembly.SetFromNodeLabels(name="NodeLB", nodeLabels=(
        ("FinalRVE", (NodeLB[0].label,)),), unsorted=True)
    # a.Set(name='NodeLB', nodes=(NodeLB,))
    NodeRB = part.sets["VertexRB"].nodes
    model_assembly.SetFromNodeLabels(name="NodeRB", nodeLabels=(
        ("FinalRVE", (NodeRB[0].label,)),), unsorted=True)
    # a.Set(name='NodeRB', nodes=(NodeRB,))
    NodeLT = part.sets["VertexLT"].nodes
    model_assembly.SetFromNodeLabels(name="NodeLT", nodeLabels=(
        ("FinalRVE", (NodeLT[0].label,)),), unsorted=True)
    # a.Set(name='NodeLT', nodes=(NodeLT,))
    NodeRT = part.sets["VertexRT"].nodes
    # a.Set(name='NodeRT', nodes=(NodeRT,))
    model_assembly.SetFromNodeLabels(name="NodeRT", nodeLabels=(
        ("FinalRVE", (NodeRT[0].label,)),), unsorted=True)

    # for Vertices left_bottom and right_upper
    model.Equation(
        name="LB_RT_1", terms=((1, "NodeRT", 1), (-1, "NodeLB", 1), (-1 * Lx, "Ref-R", 1), (-1 * Ly, "Ref-T", 1))
    )
    model.Equation(
        name="LB_RT_2", terms=((1, "NodeRT", 2), (-1, "NodeLB", 2), (-1 * Lx, "Ref-R", 2), (-1 * Ly, "Ref-T", 2))
    )

    model.Equation(
        name="LT_RB_1", terms=((1, "NodeRB", 1), (-1, "NodeLT", 1), (-1 * Lx, "Ref-R", 1), (1 * Ly, "Ref-T", 1))
    )
    model.Equation(
        name="LT_RB_2", terms=((1, "NodeRB", 2), (-1, "NodeLT", 2), (-1 * Lx, "Ref-R", 2), (1 * Ly, "Ref-T", 2))
    )

    # define the equations for the left and right edges
    if len(RightEdge_nodes_sorted) == len(LeftEdge_nodes_sorted):
        for ii in range(1, len(RightEdge_nodes_sorted) - 1):
            model_assembly.SetFromNodeLabels(
                name="LEFT_" + str(ii),
                nodeLabels=(
                    ("FinalRVE", tuple([LeftEdge_nodes_sorted[ii].label])),),
                unsorted=True,
            )
            model_assembly.SetFromNodeLabels(
                name="RIGHT_" + str(ii),
                nodeLabels=(
                    ("FinalRVE", tuple([RightEdge_nodes_sorted[ii].label])),),
                unsorted=True,
            )
            for jj in range(1, 3):
                model.Equation(
                    name="LEFT_RIGHT_" + str(ii) + "_" + str(jj),
                    terms=((1, "RIGHT_" + str(ii), jj), (-1, "LEFT_" +
                           str(ii), jj), (-1 * Lx, "Ref-R", jj)),
                )
    else:
        print("the number of nodes between the two sides are not the same")

    # part 1: equations for edges 2 (edgesFRONT_RIGHT) and 4 (edgesBACK_LEFT)

    if len(TopEdge_nodes_sorted) == len(BotEdge_nodes_sorted):
        for ii in range(1, len(TopEdge_nodes_sorted) - 1):
            model_assembly.SetFromNodeLabels(
                name="TOP_" + str(ii),
                nodeLabels=(
                    ("FinalRVE", tuple([TopEdge_nodes_sorted[ii].label])),),
                unsorted=True,
            )
            model_assembly.SetFromNodeLabels(
                name="BOT_" + str(ii),
                nodeLabels=(
                    ("FinalRVE", tuple([BotEdge_nodes_sorted[ii].label])),),
                unsorted=True,
            )
            for jj in range(1, 3):
                model.Equation(
                    name="TOP_BOT_" + str(ii) + "_" + str(jj),
                    terms=((1, "TOP_" + str(ii), jj), (-1, "BOT_" +
                           str(ii), jj), (-1 * Ly, "Ref-T", jj)),
                )
    else:
        print("the number of nodes between the two sides are not the same")

    # create step
    model.StaticStep(
        name="Step-1",
        previous="Initial",
        stabilizationMagnitude=0.0002,
        stabilizationMethod=DISSIPATED_ENERGY_FRACTION,
        continueDampingFactors=False,
        adaptiveDampingRatio=None,
        initialInc=0.02,
        maxInc=0.02,
        nlgeom=ON,
    )
    model.Temperature(
        name="Predefined Field-1",
        createStepName="Initial",
        region=model_assembly.instances["FinalRVE"].sets["Phase_1"],
        distributionType=UNIFORM,
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
        magnitudes=(120.0,),
    )
    # create Final-outputs
    model.fieldOutputRequests["F-Output-1"].setValues(
        variables=("S", "E", "LE", "ENER", "ELEN", "ELEDEN", "EVOL", "IVOL"), timeInterval=0.1
    )
    model.FieldOutputRequest(
        name="F-Output-2", createStepName="Step-1", variables=("U", "RF"), timeInterval=0.1)
    model.historyOutputRequests["H-Output-1"].setValues(
        variables=("ALLAE", "ALLCD", "ALLIE", "ALLKE", "ALLPD", "ALLSE", "ALLWK", "ETOTAL"), timeInterval=0.1
    )

    model.SmoothStepAmplitude(
        name="Amp-1", timeSpan=STEP, data=((0.0, 0.0), (1.0, 1.0)))

    # create loads
    # adding the macro strain to the
    model.DisplacementBC(
        name="E_11",
        createStepName="Step-1",
        region=model_assembly.sets["Ref-R"],
        u1=0.725238,
        u2=UNSET,
        ur3=UNSET,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None,
    )
    model.DisplacementBC(
        name="E_12",
        createStepName="Step-1",
        region=model_assembly.sets["Ref-R"],
        u1=UNSET,
        u2=0.153468,
        ur3=UNSET,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None,
    )
    model.DisplacementBC(
        name="E_21",
        createStepName="Step-1",
        region=model_assembly.sets["Ref-T"],
        u1=0.153468,
        u2=UNSET,
        ur3=UNSET,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None,
    )
    model.DisplacementBC(
        name="E_22",
        createStepName="Step-1",
        region=model_assembly.sets["Ref-T"],
        u1=UNSET,
        u2=-0.118837,
        ur3=UNSET,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName="",
        localCsys=None,
    )
    model.boundaryConditions["E_11"].setValues(amplitude="Amp-1")
    model.boundaryConditions["E_12"].setValues(amplitude="Amp-1")
    model.boundaryConditions["E_21"].setValues(amplitude="Amp-1")
    model.boundaryConditions["E_22"].setValues(amplitude="Amp-1")

    # create a job
    mdb.Job(
        name=job_name,
        model="Model-1",
        description="",
        type=ANALYSIS,
        atTime=None,
        waitMinutes=0,
        waitHours=0,
        queue=None,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        userSubroutine="",
        scratch="",
        resultsFormat=ODB,
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numGPUs=0,
    )
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
