'''
Created on 2020-04-06 17:53:59
Last modified on 2020-11-17 11:35:32

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
                             TOP_SURFACE, MIDDLE_SURFACE, FROM_SECTION)
import section

# third-party
import numpy as np


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
