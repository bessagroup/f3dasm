# Abaqus/CAE script
# Created by M.A. Bessa (M.A.Bessa@tudelft.nl) on 12-Nov-2019 06:40:20
#
from abaqus import *
from abaqusConstants import *
session.viewports['Viewport: 1'].makeCurrent()
#session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
Mdb()
#
import numpy

#------------------------------------------------------------
os.chdir(r'/home/gkus/F3DAS-master/3_Analyses/DOE_Ix-PD-100/Input_point1/Imperfection_point1/DoE_point92')
#
#-------------------------------------------------------------
# Parameters:
VertexPolygon = 3 # Number of vertices (sides) of the polygon base
power = 1.00000e+00 # Power law exponent establishing the evolution of the spacing between battens
MastDiameter = 1.00000e+02 # Radius of the circumscribing circle of the polygon
nStories = 1 # Number of stories in HALF of the strut (i.e. in a single AstroMast!)
MastPitch = 5.93607e+01 # Pitch length of the strut (i.e. a single AstroMast!)
pinned_joints = 1 # (1 = batten are pinned to longerons, 0 = battens and longerons are a solid piece)
Longeron_CS = 1.00005e+01 # (Cross Section of the longeron)
Ix = 2.86534e+01 # (Second moment of area around X axis )
Iy = 7.50000e+01 # (Second moment of area around Y axis )
J = 2.50000e+02 # (Second moment of area around X axis )
Emodulus = 1.82600e+03 # (Youngus Modulus)
Gmodulus = 6.57372e+02 # (Shear Modulus)
nu = 3.88864e-01 # (Poisson Ratio)
ConeSlope = 5.00000e-01 # Slope of the longerons (0 = straight, <0 larger at the top, >0 larger at the bottom)
Twist_angle = 0.00000e+00 # Do you want to twist the longerons?
transition_length_ratio = 1.00000e+00 # Transition zone for the longerons
#------------------------------------------------------------

MastRadius = MastDiameter/2.0
MastHeight = nStories*MastPitch

Mesh_size = min(MastRadius,MastPitch)/300.0

session.viewports['Viewport: 1'].setValues(displayedObject=None)


# Create all the joints of the a single Deployable Mast:
joints = numpy.zeros((nStories+1,VertexPolygon,3))
joints_outter = numpy.zeros((nStories+1,VertexPolygon,3))
for iStorey in range(0,nStories+1,1):
    for iVertex in range(0,VertexPolygon,1):
        # Constant spacing between each storey (linear evolution):
        Zcoord = MastHeight/nStories*iStorey
        # Power-law spacing between each storey (more frequent at the fixed end):
        #        Zcoord = MastHeight*(float(iStorey)/float(nStories))**power
        # Power-law spacing between each storey (more frequent at the rotating end):
        #        Zcoord = -MastHeight/(float(nStories)**power)*(float(nStories-iStorey)**power)+MastHeight
        # Exponential spacing between each storey
        #        Zcoord =(MastHeight+1.0)/exp(float(nStories))*exp(float(iStorey))
        #
        Xcoord = MastRadius*cos(2.0*pi/VertexPolygon*iVertex + Twist_angle*min(Zcoord/MastHeight/transition_length_ratio,1.0))
        Ycoord = MastRadius*sin(2.0*pi/VertexPolygon*iVertex + Twist_angle*min(Zcoord/MastHeight/transition_length_ratio,1.0))
        # Save point defining this joint:
        joints[iStorey,iVertex,:] = (Xcoord*(1.0-min(Zcoord,transition_length_ratio*MastHeight)/MastHeight*ConeSlope),Ycoord*(1.0-min(Zcoord,transition_length_ratio*MastHeight)/MastHeight*ConeSlope),Zcoord)
        #
        center = (0.0,0.0)
        vec = joints[iStorey,iVertex,0:2]-center
        norm_vec = numpy.linalg.norm(vec)
        joints_outter[iStorey,iVertex,2] = joints[iStorey,iVertex,2]
        joints_outter[iStorey,iVertex,0:2] = joints[iStorey,iVertex,0:2]
    # end iSide loop
    
#end iStorey loop

# Create the longerons:

p_longerons = mdb.models['Model-1'].Part(name='longerons', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)

p_longerons = mdb.models['Model-1'].parts['longerons']
session.viewports['Viewport: 1'].setValues(displayedObject=p_longerons)

d_longerons, r_longerons = p_longerons.datums, p_longerons.referencePoints

LocalDatum_list = [] # List with local coordinate system for each longeron
long_midpoints = [] # List with midpoints of longerons (just to determine a set containing the longerons)
e_long = p_longerons.edges


for iVertex in range(0,VertexPolygon,1):
    # First create local coordinate system (useful for future constraints, etc.):
    iStorey=0
    origin = joints[iStorey,iVertex,:]
    point2 = joints[iStorey,iVertex-1,:]
    name = 'Local_Datum_'+str(iVertex)
    LocalDatum_list.append(p_longerons.DatumCsysByThreePoints(origin=origin, point2=point2, name=name, 
        coordSysType=CARTESIAN, point1=(0.0, 0.0, 0.0)))
    #
    # Then, create the longerons
    templist = [] # List that will contain the points used to make each longeron
    for iStorey in range(0,nStories+1,1):
        templist.append(joints[iStorey,iVertex,:])
        if iStorey != 0: # Save midpoints of bars
            long_midpoints.append( [(joints[iStorey-1,iVertex,:]+joints[iStorey,iVertex,:])/2 , ])
        # end if
    # end iStorey loop
    p_longerons.WirePolyLine(points=templist,
        mergeType=IMPRINT, meshable=ON)
    # Create set for each longeron (to assign local beam directions)
    for i in range(0,len(templist)): # loop over longerons edges
        if i == 0:
            select_edges = e_long.findAt([templist[0], ]) # Find the first edge
        else:
            # Now find remaining edges in longerons
            temp = e_long.findAt([templist[i], ])
            select_edges = select_edges + temp
        #end if
    #end i loop
    longeron_name = 'longeron-'+str(iVertex)+'_set'
    p_longerons.Set(edges=select_edges, name=longeron_name)

#end for iVertex loop

# Longerons set:
e_long = p_longerons.edges
select_edges = []
for i in range(0,len(long_midpoints)): # loop over longerons edges
    if i == 0:
        select_edges = e_long.findAt(long_midpoints[0]) # Find the first edge
    else:
        # Now find remaining edges in longerons
        temp = e_long.findAt(long_midpoints[i])
        select_edges = select_edges + temp
    #end if

#end i loop

p_longerons.Set(edges=select_edges, name='all_longerons_set')
all_longerons_set_edges = select_edges

p_longerons.Surface(circumEdges=all_longerons_set_edges, name='all_longerons_surface')


# Create a set with all the joints:
v_long = p_longerons.vertices
select_vertices = []
select_top_vertices = []
select_bot_vertices = []
for iStorey in range(0,nStories+1,1):
    for iVertex in range(0,VertexPolygon,1):
        # Select all the joints in the longerons:
        current_joint = v_long.findAt( [joints[iStorey,iVertex,:] , ] ) # Find the first vertex
        current_joint_name = 'joint-'+str(iStorey)+'-'+str(iVertex)
        # Create a set for each joint:
        p_longerons.Set(vertices=current_joint, name=current_joint_name)
        #
        if iStorey == 0 and iVertex == 0:
            select_vertices = current_joint # Instantiate the first point in set
        else:
            select_vertices = select_vertices + current_joint # Instantiate the first point in set
        # endif iStorey == 0 and iVertex == 0
        #
        if iStorey == 0: # Also save the bottom nodes separately
            if iVertex == 0:
                # Start selecting the bottom joints for implementing the boundary conditions
                select_bot_vertices = current_joint
            else:
                select_bot_vertices = select_bot_vertices + current_joint
            # endif iStorey == 0:
        elif iStorey == nStories: # Also save the top nodes separately
            if iVertex == 0:
                # Start selecting the top joints for implementing the boundary conditions
                select_top_vertices = current_joint
            else: # remaining vertices:
                select_top_vertices = select_top_vertices + current_joint
        #end if
    #end iVertex loop

#end iStorey loop

p_longerons.Set(vertices=select_vertices, name='all_joints_set')
p_longerons.Set(vertices=select_bot_vertices, name='bot_joints_set')
p_longerons.Set(vertices=select_top_vertices, name='top_joints_set')

#
# Create materials:
mdb.models['Model-1'].Material(name='NiTi_alloy')
mdb.models['Model-1'].materials['NiTi_alloy'].Elastic(table=((83.0E3, 0.31), 
    ))
mdb.models['Model-1'].materials['NiTi_alloy'].Density(table=((1.0E-3, ), ))

mdb.models['Model-1'].Material(name='PC')
mdb.models['Model-1'].materials['PC'].Elastic(table=((2134, 0.27), 
    ))
mdb.models['Model-1'].materials['PC'].Density(table=((1.19E-3, ), ))

mdb.models['Model-1'].Material(name='PLA')
mdb.models['Model-1'].materials['PLA'].Elastic(table=((Emodulus, nu), 
    ))
mdb.models['Model-1'].materials['PLA'].Density(table=((1.24E-3, ), ))

mdb.models['Model-1'].Material(name='CNT')
mdb.models['Model-1'].materials['CNT'].Elastic(table=((1000.0E3, 0.3), 
    ))
mdb.models['Model-1'].materials['CNT'].Density(table=((1.0E-3, ), ))

# Create beam profiles and beam sections:
mdb.models['Model-1'].GeneralizedProfile(name='LongeronsProfile', area=Longeron_CS, i11=Ix, i12=0.0, i22=Iy, j=J, gammaO=0.0, gammaW=0.0)

mdb.models['Model-1'].BeamSection(name='LongeronsSection', integration=
      BEFORE_ANALYSIS, poissonRatio=0.31, beamShape=CONSTANT, 
         profile='LongeronsProfile', density=0.00124, thermalExpansion=OFF, 
         temperatureDependency=OFF, dependencies=0, table=((Emodulus, Gmodulus), ), 
         alphaDamping=0.0, betaDamping=0.0, compositeDamping=0.0, centroid=(0.0, 
         0.0), shearCenter=(0.0, 0.0), consistentMassMatrix=False)

# Assign respective sections:
p_longerons.SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=
    p_longerons.sets['all_longerons_set'], 
    sectionName='LongeronsSection', thicknessAssignment=FROM_SECTION)

# Assing beam orientation:
for iVertex in range(0,VertexPolygon,1):
    iStorey=0
    dir_vec_n1 = joints[iStorey,iVertex,:]-(0.,0.,0.) # Vector n1 perpendicular to the longeron tangent
    longeron_name = 'longeron-'+str(iVertex)+'_set'
    region=p_longerons.sets[longeron_name]
    p_longerons.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=dir_vec_n1)

#end for iVertex
#

delta = Mesh_size/100.0
########################################################################
#Mesh the structure

#refPlane = p_longerons.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=L/2)
#d = p.datums
#All_faces = facesLeafs+facesDoubleThickBoom
#p.PartitionFaceByDatumPlane(datumPlane=d[refPlane.id], faces=All_faces)
##
#session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF
#    engineeringFeatures=OFF, mesh=ON)
#session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
#    meshTechnique=ON)
#p = mdb.models['Model-1'].parts['reducedCF_TRAC_boom']

p_longerons.seedPart(size=Mesh_size, deviationFactor=0.04, minSizeFactor=0.001,
    constraint=FINER)
p_longerons.seedEdgeBySize(edges=all_longerons_set_edges, size=Mesh_size, deviationFactor=0.04,
    constraint=FINER)
elemType_longerons = mesh.ElemType(elemCode=B31, elemLibrary=STANDARD) # Element type
p_longerons.setElementType(regions=(all_longerons_set_edges, ), elemTypes=(elemType_longerons, ))
p_longerons.generateMesh()

#######################################################################

# Make Analytical surfaces for contact purposes
s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=MastRadius*3.0)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
s1.Line(point1=(0.0, -MastRadius*1.1), point2=(0.0, MastRadius*1.1))
s1.VerticalConstraint(entity=g[2], addUndoState=False)
p_surf = mdb.models['Model-1'].Part(name='AnalyticSurf', dimensionality=THREE_D, 
    type=ANALYTIC_RIGID_SURFACE)
p_surf = mdb.models['Model-1'].parts['AnalyticSurf']
p_surf.AnalyticRigidSurfExtrude(sketch=s1, depth=MastRadius*2.2)
s1.unsetPrimaryObject()

rigid_face = p_surf.faces
#surf_select = f.findAt((0.0,MastRadius*1.05,0.0))
#surf_select = f[0]
p_surf.Surface(side1Faces=rigid_face, name='rigid_support')
#p_surf.Set(faces=surf_select, name='support_surface_set')
#p_surf.sets['all_diagonals_set']

#
# Make assembly:
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
# Create reference points to assign boundary conditions
RP_ZmYmXm = a.ReferencePoint(point=(0.0, 0.0, -1.1*MastRadius))
refpoint_ZmYmXm = (a.referencePoints[RP_ZmYmXm.id],)
a.Set(referencePoints=refpoint_ZmYmXm, name='RP_ZmYmXm')
#
RP_ZpYmXm = a.ReferencePoint(point=(0.0, 0.0, MastHeight+1.1*MastRadius))
refpoint_ZpYmXm = (a.referencePoints[RP_ZpYmXm.id],)
a.Set(referencePoints=refpoint_ZpYmXm, name='RP_ZpYmXm')
#
# Create longerons
a_long = a.Instance(name='longerons-1-1', part=p_longerons, dependent=ON)
# Create bottom surface
a_surf_bot = a.Instance(name='AnalyticSurf-1-1', part=p_surf, dependent=ON)
# Now rotate the plane to have the proper direction
a.rotate(instanceList=('AnalyticSurf-1-1', ), axisPoint=(0.0, 0.0, 0.0), 
    axisDirection=(0.0, 1.0, 0.0), angle=90.0)
#
# Create set with surface
select_bot_surf=a_surf_bot.surfaces['rigid_support']
# Perhaps we need to define a set instead of a face
#AnalyticSurf_surface=a_surf_bot.Surface(side1Faces=select_bot_surf, name='support_surf_bot-1')
mdb.models['Model-1'].RigidBody(name='Constraint-RigidBody_surf_bot-1', refPointRegion=refpoint_ZmYmXm, 
    surfaceRegion=select_bot_surf)
for iVertex in range(0,VertexPolygon,1):
    #
    # Select appropriate coordinate system:
    DatumID = LocalDatum_list[iVertex].id
    datum = a_long.datums[DatumID]
    for iStorey in range(0,nStories+1,1):
        # Current joint:
        current_joint_name = 'joint-'+str(iStorey)+'-'+str(iVertex)
        # Define COUPLING constraints for all the joints:
        if iStorey == 0: # Bottom base:
            #
            master_region=a.sets['RP_ZmYmXm'] # Note that the master is the Reference Point
            #
            slave_region=a_long.sets[current_joint_name]
            # Make constraint for this joint:
            Constraint_name = 'RP_ZmYmXm_PinConstraint-'+str(iStorey)+'-'+str(iVertex)
            mdb.models['Model-1'].Coupling(name=Constraint_name, controlPoint=master_region, 
                surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
                localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)
            #
            #Constraint_name = 'RP_ZmYmXm_FixedConstraint-'+str(iStorey)+'-'+str(iVertex)
            #mdb.models['Model-1'].Coupling(name=Constraint_name, controlPoint=master_region, 
            #    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
            #    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
            # Make constraint for this joint:
        elif iStorey == nStories: # Top base:
            #
            master_region=a.sets['RP_ZpYmXm'] # Note that the master is the Reference Point
            #
            slave_region=a_long.sets[current_joint_name]
            # Make constraint for this joint:
            Constraint_name = 'RP_ZpYmXm_PinConstraint-'+str(iStorey)+'-'+str(iVertex)
            mdb.models['Model-1'].Coupling(name=Constraint_name, controlPoint=master_region, 
                surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
                localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)
            #
            #Constraint_name = 'RP_ZpYmXm_FixedConstraint-'+str(iStorey)+'-'+str(iVertex)
            #mdb.models['Model-1'].Coupling(name=Constraint_name, controlPoint=master_region, 
            #    surface=slave_region, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
            #    localCsys=datum, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
            # Make constraint for this joint:
        else: # Middle stories:
            master_region=a_long.sets[current_joint_name]
            #
            slave_region=a_bat.sets[current_joint_name]
            # Make constraint for this joint:
        #endif iStorey
        #
    #end for iStorey

#end for iVertex


#

# Create hinges:
#select_joints=a.instances['deployable_mast-1'].sets['all_joints_set']
#select_RefPoint=a.sets['RP_joints']
#mdb.models['Model-1'].RigidBody(name='JointsContraint', refPointRegion=select_RefPoint, 
#    pinRegion=select_joints)

#
# Export mesh to .inp file
#
mdb.Job(name='include_mesh_DoE92', model='Model-1', type=ANALYSIS, explicitPrecision=SINGLE,
    nodalOutputPrecision=SINGLE, description='',
    parallelizationMethodExplicit=DOMAIN, multiprocessingMode=DEFAULT,
    numDomains=1, userSubroutine='', numCpus=1, memory=90,
    memoryUnits=PERCENTAGE, scratch='', echoPrint=OFF, modelPrint=OFF,
    contactPrint=OFF, historyPrint=OFF)
import os
mdb.jobs['include_mesh_DoE92'].writeInput(consistencyChecking=OFF)
# End of python script

