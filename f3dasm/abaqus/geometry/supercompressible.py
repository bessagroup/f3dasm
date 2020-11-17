'''
Created on 2020-11-17 11:33:18
Last modified on 2020-11-17 11:45:23

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


# imports

from __future__ import division

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (THREE_D, DEFORMABLE_BODY, ON, OFF,
                             WHOLE_SURFACE, KINEMATIC, MIDDLE_SURFACE,
                             FROM_SECTION, CARTESIAN, IMPRINT, CONSTANT,
                             BEFORE_ANALYSIS, N1_COSINES, B31, FINER,
                             ANALYTIC_RIGID_SURFACE, LINEAR, DURING_ANALYSIS)
from part import EdgeArray
import mesh


# standard library
import itertools

# third-party
import numpy as np

# local library
from .base import Geometry
from ..material.abaqus_materials import AbaqusMaterial


# TODO: review under rve development strategy
# TODO: apply strategy pattern


# Bessa, 2019 (supercompressible metamaterial)

class Supercompressible(Geometry):

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
        # TODO: create outside?
        AbaqusMaterial(name=material_name, props=props, create_section=False,
                       model=model)

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
