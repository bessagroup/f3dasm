'''
Created on 2020-04-06 17:53:59
Last modified on 2020-11-02 12:07:44

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create structures that can be used within models.

Notes
-----
-Based on code developed by M. A. Bessa.

References
----------
1. Bessa, M. A. and S. Pellegrino (2018) International Journal of Solids and
Structures 139-140: 174-188.
'''


# imports

from __future__ import division

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (CLOCKWISE, THREE_D, DEFORMABLE_BODY, ON, OFF,
                             DISCRETE, SURFACE, AXIS_1, AXIS_3, EDGE,
                             WHOLE_SURFACE, KINEMATIC, XYPLANE, UNIFORM,
                             GRADIENT, NO_IDEALIZATION, DEFAULT, SIMPSON,
                             TOP_SURFACE, MIDDLE_SURFACE, FROM_SECTION,
                             CARTESIAN, IMPRINT, CONSTANT, BEFORE_ANALYSIS,
                             N1_COSINES, B31, FINER, ANALYTIC_RIGID_SURFACE,
                             LINEAR, DURING_ANALYSIS)
import section
from part import EdgeArray
import mesh


# standard library
import itertools

# third-party
import numpy as np

# local library
from ..material.abaqus_materials import AbaqusMaterial


# TODO: create general object and move this structures outside

# Bessa, 2018 (TRAC boom)

class TRACBoom(object):

    def __init__(self, height, radius, theta, thickness, length, material,
                 name='TRACBOOM', layup=None, rotation_axis=1):
        '''
        Parameters
        ----------
        height : float
            Connection between structure and TRAC boom.
        radius : float
            Radius of curvature for the ends of structure.
        theta : float
            Angle of curved ends of structure (degrees).
        thickness : float
            Shell thickness.
        length : float
            Length of the structure (z-direction)
        material : instance of any abaqus_materials module class
            Material of the ply.
        layup : array-like
            List of angles (degrees) from concave to convex face. It assumes
            both leafs have the same material and layup. If None, then it
            considers an HomogeneousShellSection.
        rotation_axis : int. possible values = 1 or 2
            Axis about which moment will be applied.
        '''
        self.height = height
        self.radius = radius
        self.theta = theta * np.pi / 180.
        self.thickness = thickness
        self.length = length
        self.material = material
        self.layup = layup
        self.rotation_axis = rotation_axis
        self.name = name
        # mesh definitions
        self.mesh_size = min(1.0e-3, (self.theta * self.radius + self.height) / 30.)
        self.mesh_deviation_factor = .4
        self.mesh_min_size_factor = .4
        # additional variables
        self.ref_point_positions = ['MINUS', 'CENTER', 'PLUS']

    def change_mesh_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_part(self, model):

        # create geometry
        part = self._create_geometry(model)

        # create required sets and surfaces
        self._create_sets(part)

        # assign material and material orientation
        self._assign_material(model, part)
        self._create_material_orientation(part)

        # make partitions for meshing purposes
        self._make_partitions(part)

        # generate mesh
        self._generate_mesh(part)

    def create_instance(self, model):

        # initialization
        part = model.parts[self.name]

        # create instance
        model.rootAssembly.Instance(name=self.name, part=part,
                                    dependent=ON)

        # create reference points for boundary conditions
        self._create_ref_points(model)

        # add constraints for loading
        self._create_coupling_constraints(model)

    def _create_geometry(self, model):

        # initialization
        gamma = np.pi / 2 - self.theta

        # create sketch
        sheet_size = 3 * (2 * self.height + 2 * self.radius)
        s = model.ConstrainedSketch(name=self.name + '_PROFILE',
                                    sheetSize=sheet_size)

        # draw on sketch
        s.Line(point1=(-self.height, 0.),
               point2=(0., 0.))  # connection region (thickness 2*t)
        s.ArcByCenterEnds(
            center=(-self.height, self.radius),
            point1=(-self.height, 0.0),
            point2=(-self.height - self.radius * np.cos(gamma), self.radius * (1 - np.sin(gamma))),
            direction=CLOCKWISE)  # upper leaf of boom
        s.ArcByCenterEnds(
            center=(-self.height, -self.radius),
            point1=(-self.height - self.radius * np.cos(gamma), self.radius * (-1 + np.sin(gamma))),
            point2=(-self.height, 0.0), direction=CLOCKWISE)  # lower leaf of boom

        # extrude sketch
        part = model.Part(name=self.name, dimensionality=THREE_D,
                          type=DEFORMABLE_BODY)
        part.BaseShellExtrude(sketch=s, depth=self.length)

        return part

    def _create_sets(self, part):

        # initialization
        delta = self.mesh_size / 10.
        gamma = np.pi / 2 - self.theta

        # edges
        e = part.edges
        # all edges in Z plus
        edges = e.getByBoundingBox(-self.radius - self.height - delta, -2 * self.radius - delta,
                                   self.length - delta, self.radius + self.height + delta,
                                   2 * self.radius + delta, self.length + delta)
        part.Set(edges=edges, name='ZPLUS_EDGES')
        # all edges in Z minus
        edges = e.getByBoundingBox(-self.radius - self.height - delta, -2 * self.radius - delta,
                                   - delta, self.radius + self.height + delta, 2 * self.radius + delta,
                                   delta)
        part.Set(edges=edges, name='ZMINUS_EDGES')
        # edge of the upper leaf
        edges = e.getByBoundingBox(-self.radius * np.cos(gamma) - self.height - delta,
                                   self.radius * (1 - np.sin(gamma)) - delta, - delta,
                                   -self.radius * np.cos(gamma) - self.height + delta,
                                   self.radius * (1 - np.sin(gamma)) + delta,
                                   self.length + delta)
        part.Set(edges=edges, name='UPPER_LEAF_EDGE')
        # edge of the lower leaf
        edges = e.getByBoundingBox(-self.radius * np.cos(gamma) - self.height - delta,
                                   self.radius * (-1 + np.sin(gamma)) - delta, - delta,
                                   -self.radius * np.cos(gamma) - self.height + delta,
                                   self.radius * (-1 + np.sin(gamma)) + delta, self.length + delta)
        part.Set(edges=edges, name='LOWER_LEAF_EDGE')

        # faces
        f = part.faces
        # upper and lower leafs
        pt1 = [-self.height, self.radius, -delta]
        pt2 = [-self.height, self.radius, self.length + delta]
        # upper
        upperLeaf = f.getByBoundingCylinder(pt1, pt2, self.radius + delta)
        part.Set(faces=upperLeaf, name='UPPER_LEAF_FACE')
        part.Surface(side2Faces=upperLeaf, name='UPPER_LEAF_CONCAVE_SURF')
        part.Surface(side1Faces=upperLeaf, name='UPPER_LEAF_CONVEX_SURF')
        # lower
        pt1[1] = pt2[1] = -self.radius
        lowerLeaf = f.getByBoundingCylinder(pt1, pt2, self.radius + delta)
        part.Set(faces=lowerLeaf, name='LOWER_LEAF_FACE')
        part.Surface(side2Faces=lowerLeaf, name='LOWER_LEAF_CONCAVE_SURF')
        part.Surface(side1Faces=lowerLeaf, name='LOWER_LEAF_CONVEX_SURF')
        # both
        facesLeafs = upperLeaf + lowerLeaf
        part.Set(faces=facesLeafs, name='LEAFS_FACES')
        # double laminate
        doubleLaminate = f.getByBoundingBox(-self.height - delta, -delta, - delta,
                                            delta, delta,
                                            self.length + delta)
        part.Set(faces=doubleLaminate, name='DOUBLE_LAMINATE_FACE')
        part.Surface(side2Faces=doubleLaminate, name='DOUBLE_LAMINATE_TOP_SURF')
        part.Surface(side1Faces=doubleLaminate, name='DOUBLE_LAMINATE_BOTTOM_SURF')

    def _create_ref_points(self, model):

        # initialization
        modelAssembly = model.rootAssembly

        # create reference points
        for i, position in enumerate(self.ref_point_positions):
            rp = modelAssembly.ReferencePoint(point=(0.0, 0.0, i * self.length / 2))
            modelAssembly.Set(referencePoints=(modelAssembly.referencePoints[rp.id],),
                              name=self._get_ref_point_name(position))

        # add equation to relate the reference points
        dof = 3 + self.rotation_axis
        model.Equation(name='RELATE_RPS',
                       terms=((1.0, self._get_ref_point_name(self.ref_point_positions[0]), dof),
                              (1.0, self._get_ref_point_name(self.ref_point_positions[-1]), dof)))

    def _create_coupling_constraints(self, model):

        # create coupling constraints
        for position in self.ref_point_positions[::2]:
            region1 = model.rootAssembly.sets[self._get_ref_point_name(position)]
            region2 = model.rootAssembly.instances[self.name].sets['Z%s_EDGES' % position]

            model.Coupling(name='CONSTRAINT-Z%s' % position, controlPoint=region1,
                           surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC,
                           localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    def _create_material_orientation(self, part):

        # define regions
        regions = ['UPPER_LEAF_FACE', 'LOWER_LEAF_FACE', 'DOUBLE_LAMINATE_FACE']
        normal_axis_regions = ['UPPER_LEAF_CONVEX_SURF', 'LOWER_LEAF_CONVEX_SURF',
                               'DOUBLE_LAMINATE_BOTTOM_SURF']
        primary_axis_regions = ['UPPER_LEAF_EDGE', 'LOWER_LEAF_EDGE', 'LOWER_LEAF_EDGE']

        # create material orientations
        for region_name, normalAxisRegion_name, primaryAxisRegion_name in zip(
                regions, normal_axis_regions, primary_axis_regions):

            region = part.sets[region_name]
            normalAxisRegion = part.surfaces[normalAxisRegion_name]
            primaryAxisRegion = part.sets[primaryAxisRegion_name]

            part.MaterialOrientation(
                region=region, orientationType=DISCRETE, axis=AXIS_3,
                normalAxisDefinition=SURFACE, normalAxisRegion=normalAxisRegion,
                flipNormalDirection=False, normalAxisDirection=AXIS_3,
                primaryAxisDefinition=EDGE, primaryAxisRegion=primaryAxisRegion,
                primaryAxisDirection=AXIS_1, flipPrimaryDirection=False)

    def _make_partitions(self, part):

        # create reference plane
        refPlane = part.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE,
                                                   offset=self.length / 2.)

        # make partition
        part.PartitionFaceByDatumPlane(datumPlane=part.datums[refPlane.id],
                                       faces=part.faces)

    def _generate_mesh(self, part):

        # seed part
        part.seedPart(size=self.mesh_size,
                      deviationFactor=self.mesh_deviation_factor,
                      minSizeFactor=self.mesh_min_size_factor)

        # generate mesh
        part.generateMesh()

    def _assign_material(self, model, part):

        # initialization
        section_names = ['SINGLE_LAMINATE', 'DOUBLE_LAMINATE']
        regions = ['LEAFS_FACES', 'DOUBLE_LAMINATE_FACE']
        offset_types = [TOP_SURFACE, MIDDLE_SURFACE]

        # create sections
        if self.layup is None:
            self._create_homogeneous_sections(model, section_names)
        else:
            self._create_composite_sections(model, section_names)

        # assign sections
        for name, region, offset_type in zip(section_names, regions, offset_types):
            region_ = part.sets[region]
            part.SectionAssignment(
                region=region_, sectionName=name, offset=0.0,
                offsetType=offset_type, offsetField='',
                thicknessAssignment=FROM_SECTION)

    def _create_homogeneous_sections(self, model, names):

        for i, name in enumerate(names):
            j = i + 1
            model.HomogeneousShellSection(
                name, material=self.material.name, thicknessType=UNIFORM,
                thickness=j * self.thickness)

    def _create_composite_sections(self, model, names):

        # create composite plies
        plies = self._create_composite_plies()

        # create composite shells sections
        for name, layup in zip(names, plies):
            model.CompositeShellSection(
                name=name, preIntegrate=OFF, idealization=NO_IDEALIZATION,
                symmetric=False, thicknessType=UNIFORM, poissonDefinition=DEFAULT,
                thicknessModulus=None, temperature=GRADIENT, useDensity=OFF,
                integrationRule=SIMPSON, layup=layup)

        return names

    def _create_composite_plies(self):

        # initialization
        layup = list(self.layup)
        ply_thickness = self.thickness / len(layup)

        # leafs
        leaf_composite_plies = [self._define_ply('LEAF_PLY_', i, orientation, ply_thickness)
                                for i, orientation in enumerate(layup)]

        # double
        double_composite_plies = [self._define_ply('DOUBLE_PLY_', i, orientation, ply_thickness)
                                  for i, orientation in enumerate(layup + layup[::-1])]

        return leaf_composite_plies, double_composite_plies

    def _define_ply(self, ply_name, i, orientation, ply_thickness):
        sec = section.SectionLayer(
            material=self.material.name, thickness=ply_thickness,
            orientAngle=orientation, numIntPts=3,
            plyName='%s_%i' % (ply_name, i))

        return sec

    def _get_ref_point_name(self, position):

        return 'Z%s_REF_POINT' % position


# Bessa, 2019 (supercompressible metamaterial)

class Supercompressible(object):

    def __init__(self, n_longerons, bottom_diameter, top_diameter, pitch,
                 young_modulus, shear_modulus, cross_section_props,
                 twist_angle=0., transition_length_ratio=1., n_storeys=1,
                 z_spacing='uni', power=1., name='SUPERCOMPRESSIBLE'):
        '''
        Parameters
        ----------
        n_longerons : int
            Number of longerons.
        bottom_diameter : float
            Radius of the cicumscribing circle of the polygon (bottom).
        top_diameter : float
            Radius of the cicumscribing circle of the polygon (top).
        pitch : float
            Pitch length of the structure.
        cross_section_props : dict
            Stores the information about the cross-section. Specify the type
            of the cross section using 'type'. An empty 'type' will be
            understood as generalized cross section. Different types of
            sections are allowed:
                -'circular': requires 'd'
                -'generalized': requires 'Ixx', 'Iyy', 'J', 'area'
        young_modulus, shear_modulus : float
            Material properties.
        twist_angle : float
            Longerongs twisting angle.
        transition_length_ratio = float
            Transition zone for the longerons.
        n_storeys : int
            Number of stories in half of the structure.
        z_spacing : str
            How to compute spacing between storeys. Possible values are 'uni',
            'power_rotating', 'power_fixed', 'exponential'.
        power : float
            Value for the exponent of the power law if z_spacing is
            'power_rotationg' or 'power_fixed'.

        Notes
        -----
        -it is not required to create a material (as is normal for similar
        classes) because the beam section receives directly material properties.
        '''
        # store variables
        self.n_longerons = n_longerons
        self.bottom_diameter = bottom_diameter
        self.top_diameter = top_diameter
        self.pitch = pitch
        self.young_modulus = young_modulus
        self.shear_modulus = shear_modulus
        self.cross_section_props = cross_section_props
        self.twist_angle = twist_angle
        self.transition_length_ratio = transition_length_ratio
        self.n_storeys = n_storeys
        self.z_spacing = z_spacing
        self.power = power
        # initialization
        self.joints = np.zeros((self.n_storeys + 1, self.n_longerons, 3))
        self.longeron_points = []
        # computations
        self.mast_radius = self.bottom_diameter / 2.
        self.mast_height = self.n_storeys * self.pitch
        self.cone_slope = (self.bottom_diameter - self.top_diameter) / self.bottom_diameter
        # mesh definitions
        self.mesh_size = min(self.mast_radius, self.pitch) / 300.
        self.mesh_deviation_factor = .04
        self.mesh_min_size_factor = .001
        self.element_code = B31
        # additional variables
        self.longerons_name = 'LONGERONS'
        self.surface_name = 'ANALYTICAL_SURF'
        self.ref_point_positions = ['BOTTOM', 'TOP']

    def change_mesh_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_part(self, model):

        # create geometry
        part_longerons, part_surf = self._create_geometry(model)

        # create required sets and surfaces
        self._create_sets(part_longerons, part_surf)

        # create section and assign orientation
        self._create_beam_section(model, part_longerons)

        # generate mesh
        self._generate_mesh(part_longerons)

    def create_instance(self, model):

        # initialization
        part_longerons = model.parts[self.longerons_name]
        part_surf = model.parts[self.surface_name]

        # create instances
        model.rootAssembly.Instance(name=self.longerons_name,
                                    part=part_longerons, dependent=ON)
        model.rootAssembly.Instance(name=self.surface_name,
                                    part=part_surf, dependent=ON)

        # rotate surface
        model.rootAssembly.rotate(instanceList=(self.surface_name, ),
                                  axisPoint=(0., 0., 0.),
                                  axisDirection=(0., 1., 0.), angle=90.)

        # create reference points for boundary conditions
        self._create_ref_points(model)

        # add constraints for loading
        self._create_constraints(model)

    def _create_geometry(self, model):

        # create joints
        self._create_joints()

        # create longerons
        part_longerons = self._create_geometry_longerons(model)

        # create surface
        part_surf = self._create_geometry_surf(model)

        return part_longerons, part_surf

    def _create_joints(self):

        for i_storey in range(0, self.n_storeys + 1, 1):
            zcoord = self._get_zcoord(i_storey)
            aux1 = 2.0 * np.pi / self.n_longerons
            aux2 = self.twist_angle * min(zcoord / self.mast_height / self.transition_length_ratio, 1.0)
            for i_vertex in range(0, self.n_longerons):
                aux3 = aux1 * i_vertex + aux2
                xcoord = self.mast_radius * np.cos(aux3)
                ycoord = self.mast_radius * np.sin(aux3)
                self.joints[i_storey, i_vertex, :] = (xcoord * (1.0 - min(zcoord, self.transition_length_ratio * self.mast_height) / self.mast_height * self.cone_slope), ycoord * (1.0 - min(zcoord, self.transition_length_ratio * self.mast_height) / self.mast_height * self.cone_slope), zcoord)

    def _get_zcoord(self, i_storey):

        # get spacing for selected distribution
        if self.z_spacing == 'uni':
            # constant spacing between each storey (linear evolution):
            zcoord = self.mast_height / self.n_storeys * i_storey
        elif self.z_spacing == 'power_rotating':
            # power-law spacing between each storey (more frequent at the rotating end):
            zcoord = -self.mast_height / (float(self.n_storeys)**self.power) * (float(self.n_storeys - i_storey)**self.power) + self.mast_height
        elif self.z_spacing == 'power_fixed':
            # power-law spacing between each storey (more frequent at the fixed end):
            zcoord = self.mast_height * (float(i_storey) / float(self.n_storeys))**self.power
        elif self.z_spacing == 'exponential':
            # exponential spacing between each storey
            zcoord = (self.mast_height + 1.0) / np.exp(float(self.n_storeys)) * np.exp(float(i_storey))

        return zcoord

    def _create_geometry_longerons(self, model):

        # create part
        part_longerons = model.Part(self.longerons_name, dimensionality=THREE_D,
                                    type=DEFORMABLE_BODY)

        # create datum and white
        for i_vertex in range(0, self.n_longerons):

            # get required points
            self.longeron_points.append([self.joints[i_storey, i_vertex, :] for i_storey in range(0, self.n_storeys + 1)])

            # create wires
            part_longerons.WirePolyLine(points=self.longeron_points[-1],
                                        mergeType=IMPRINT, meshable=ON)

        return part_longerons

    def _create_geometry_surf(self, model):

        # create sketch
        s = model.ConstrainedSketch(name='SURFACE_SKETCH',
                                    sheetSize=self.mast_radius * 3.0)

        # draw on sketch
        s.Line(point1=(0.0, -self.mast_radius * 1.1),
               point2=(0.0, self.mast_radius * 1.1))

        # extrude sketch
        part_surf = model.Part(name=self.surface_name, dimensionality=THREE_D,
                               type=ANALYTIC_RIGID_SURFACE)
        part_surf.AnalyticRigidSurfExtrude(sketch=s,
                                           depth=self.mast_radius * 2.2)

        return part_surf

    def _create_sets(self, part_longerons, part_surf):

        # surface
        part_surf.Surface(side1Faces=part_surf.faces,
                          name=self.surface_name)

        # longeron
        edges = part_longerons.edges
        vertices = part_longerons.vertices

        # individual sets
        all_edges = []
        for i_vertex, long_pts in enumerate(self.longeron_points):

            # get vertices and edges
            selected_vertices = [vertices.findAt((pt,)) for pt in long_pts]
            all_edges.append(EdgeArray([edges.findAt(pt) for pt in long_pts]))

            # individual sets
            long_name = self._get_longeron_name(i_vertex)
            part_longerons.Set(edges=all_edges[-1], name=long_name)

            # joints
            for i_storey, vertex in enumerate(selected_vertices):
                joint_name = self._get_joint_name(i_storey, i_vertex)
                part_longerons.Set(vertices=vertex, name=joint_name)
        name = 'ALL_LONGERONS'
        part_longerons.Set(edges=all_edges, name=name)
        name = 'ALL_LONGERONS_SURF'
        part_longerons.Surface(circumEdges=all_edges, name=name)

        # joint sets
        selected_vertices = []
        for i_storey in range(0, self.n_storeys + 1):
            selected_vertices.append([])
            for i_vertex in range(0, self.n_longerons):
                name = self._get_joint_name(i_storey, i_vertex)
                selected_vertices[-1].append(part_longerons.sets[name].vertices)
        name = 'BOTTOM_JOINTS'
        part_longerons.Set(name=name, vertices=selected_vertices[0])
        name = 'TOP_JOINTS'
        part_longerons.Set(name=name, vertices=selected_vertices[-1])
        name = 'ALL_JOINTS'
        all_vertices = list(itertools.chain(*selected_vertices))
        part_longerons.Set(name=name, vertices=all_vertices)

    def _create_beam_section(self, model, part_longerons):

        # initialization
        profile_name = 'LONGERONS_PROFILE'
        section_name = 'LONGERONS_SECTION'

        # assign the right method for the creation of the beam section
        # TODO: add more particular sections
        create_beam_section = {'generalized': self._create_generalized_beam_section,
                               'circular': self._create_circular_section}

        # create profile and beam section
        create_beam_section[self.cross_section_props.get('type', 'generalized')](
            model, profile_name, section_name)

        # section assignment
        part_longerons.SectionAssignment(
            offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
            region=part_longerons.sets['ALL_LONGERONS'],
            sectionName=section_name, thicknessAssignment=FROM_SECTION)

        # section orientation
        for i_vertex, pts in enumerate(self.longeron_points):
            dir_vec_n1 = np.array(pts[0]) - (0., 0., 0.)
            longeron_name = self._get_longeron_name(i_vertex)
            region = part_longerons.sets[longeron_name]
            part_longerons.assignBeamSectionOrientation(
                region=region, method=N1_COSINES, n1=dir_vec_n1)

    def _create_circular_section(self, model, profile_name, section_name):

        # initialization
        r = self.cross_section_props['d'] / 2.

        # create material
        material_name = 'LONGERON_MATERIAL'
        props = {'E': self.young_modulus,
                 'nu': self.young_modulus / (2 * self.shear_modulus) - 1}
        AbaqusMaterial(name=material_name, props=props, create_section=False)

        # create profile
        model.CircularProfile(name=profile_name, r=r)

        # create profile
        model.BeamSection(consistentMassMatrix=False, integration=DURING_ANALYSIS,
                          material=material_name, name=section_name,
                          poissonRatio=0.31, profile=profile_name,
                          temperatureVar=LINEAR)

    def _create_generalized_beam_section(self, model, profile_name, section_name):

        # initialization
        Ixx = self.cross_section_props['Ixx']
        Iyy = self.cross_section_props['Iyy']
        area = self.cross_section_props['area']
        J = self.cross_section_props['J']

        # create profile
        model.GeneralizedProfile(name=profile_name,
                                 area=area, i11=Ixx,
                                 i12=0., i22=Iyy, j=J, gammaO=0.,
                                 gammaW=0.)

        # create section
        model.BeamSection(name=section_name, integration=BEFORE_ANALYSIS,
                          beamShape=CONSTANT, profile=profile_name, thermalExpansion=OFF,
                          temperatureDependency=OFF, dependencies=0,
                          table=((self.young_modulus, self.shear_modulus),),
                          poissonRatio=.31,
                          alphaDamping=0.0, betaDamping=0.0, compositeDamping=0.0,
                          centroid=(0.0, 0.0), shearCenter=(0.0, 0.0),
                          consistentMassMatrix=False)

    def _generate_mesh(self, part_longerons):

        # seed part
        part_longerons.seedPart(
            size=self.mesh_size, deviationFactor=self.mesh_deviation_factor,
            minSizeFactor=self.mesh_min_size_factor, constraint=FINER)

        # assign element type
        elem_type_longerons = mesh.ElemType(elemCode=self.element_code)
        part_longerons.setElementType(regions=(part_longerons.edges,),
                                      elemTypes=(elem_type_longerons,))

        # generate mesh
        part_longerons.generateMesh()

    def _create_ref_points(self, model):

        # initialization
        modelAssembly = model.rootAssembly

        # create reference points
        for i, position in enumerate(self.ref_point_positions):
            sign = 1 if i else -1
            rp = modelAssembly.ReferencePoint(
                point=(0., 0., i * self.mast_height + sign * 1.1 * self.mast_radius))
            modelAssembly.Set(referencePoints=(modelAssembly.referencePoints[rp.id],),
                              name=self._get_ref_point_name(position))

    def _create_constraints(self, model):

        # initialization
        modelAssembly = model.rootAssembly
        instance_longerons = modelAssembly.instances[self.longerons_name]
        instance_surf = modelAssembly.instances[self.surface_name]
        ref_points = [modelAssembly.sets[self._get_ref_point_name(position)]
                      for position in self.ref_point_positions]

        # bottom point and analytic surface
        surf = instance_surf.surfaces[self.surface_name]
        model.RigidBody('CONSTRAINT-RIGID_BODY-BOTTOM', refPointRegion=ref_points[0],
                        surfaceRegion=surf)

        # create local datums
        datums = self._create_local_datums(model)

        # create coupling constraints
        for i_vertex in range(self.n_longerons):
            datum = instance_longerons.datums[datums[i_vertex].id]
            for i, i_storey in enumerate([0, self.n_storeys]):
                joint_name = self._get_joint_name(i_storey, i_vertex)
                slave_region = instance_longerons.sets[joint_name]
                master_region = ref_points[i]
                constraint_name = 'CONSTRAINT-%s-%i-%i' % (self._get_ref_point_name(self.ref_point_positions[i]),
                                                           i_storey, i_vertex)
                model.Coupling(name=constraint_name, controlPoint=master_region,
                               surface=slave_region, influenceRadius=WHOLE_SURFACE,
                               couplingType=KINEMATIC, localCsys=datum, u1=ON,
                               u2=ON, u3=ON, ur1=OFF, ur2=ON, ur3=ON)

    def _create_local_datums(self, model):
        '''
        Create local coordinate system (useful for future constraints, etc.).
        '''

        # initialization
        part_longerons = model.parts[self.longerons_name]

        datums = []
        for i_vertex in range(0, self.n_longerons):

            origin = self.joints[0, i_vertex, :]
            point2 = self.joints[0, i_vertex - 1, :]
            name = self._get_local_datum_name(i_vertex)
            datums.append(part_longerons.DatumCsysByThreePoints(
                origin=origin, point2=point2, name=name, coordSysType=CARTESIAN,
                point1=(0.0, 0.0, 0.0)))

        return datums

    def _get_local_datum_name(self, i_vertex):
        return 'LOCAL_DATUM_{}'.format(i_vertex)

    def _get_longeron_name(self, i_vertex):
        return 'LONGERON-{}'.format(i_vertex)

    def _get_joint_name(self, i_storey, i_vertex):
        return 'JOINT-{}-{}'.format(i_storey, i_vertex)

    def _get_ref_point_name(self, position):
        return 'Z{}_REF_POINT'.format(position)
