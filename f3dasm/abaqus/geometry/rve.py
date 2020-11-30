'''
Created on 2020-03-24 14:33:48
Last modified on 2020-11-30 14:20:55

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create an RVE class from which more useful classes can inherit.

References
----------
1. van der Sluis, O., et al. (2000). Mechanics of Materials 32(8): 449-462.
2. Melro, A. R. (2011). PhD thesis. University of Porto.
'''

# imports
from __future__ import division

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (TWO_D_PLANAR, DEFORMABLE_BODY, ON, FIXED,
                             THREE_D, DELETE, GEOMETRY, TET, FREE,
                             YZPLANE, XZPLANE, XYPLANE, SUPPRESS)
from part import (EdgeArray, FaceArray)
from mesh import MeshNodeArray

# standard
from abc import abstractmethod
from abc import ABCMeta
from collections import OrderedDict

# local library
from .base import Geometry
from ..modelling.bcs import DisplacementBC
from ..modelling.mesh import MeshGenerator
from ...utils.linalg import symmetricize_vector
from ...utils.solid_mechanics import compute_small_strains_from_green
from ...utils.utils import unnest
from ...utils.utils import get_decimal_places


# TODO: handle warnings and errors using particular class
# TODO: base default mesh size in characteristic length
# TODO: use more polymorphic names in methods

# object definition

def _RVE_objects_initializer(name, dims, center, tol, bcs_type):
    # TODO: object factory?

    dim = len(dims)

    if bcs_type == 'periodic':
        obj_creator = PeriodicRVEObjCreator()
        if dim == 2:
            info = PeriodicRVEInfo2D(name, dims, center, tol)
            mesh = PeriodicMeshGenerator2D()
        else:
            info = PeriodicRVEInfo3D(name, dims, center, tol)
            # TODO: pass as argument the mesh generator
            mesh = PeriodicMeshGenerator3DSimple()
        bcs = PeriodicBoundaryConditions(info)

    return info, obj_creator, mesh, bcs


# TODO: inherit from geometry

class RVE(Geometry):
    __metaclass__ = ABCMeta

    def __init__(self, name, dims, center, material, tol, bcs_type):
        '''
        Parameters
        ----------
        bcs_type : str
            Possible values are 'periodic'.
        '''
        super(RVE, self).__init__(default_mesh=False)
        # variable initialization
        self.particles = []
        self.material = material
        # auxiliar variables
        self._create_instance = False
        # create objects
        self.info, self.obj_creator, self.mesh, self.bcs = _RVE_objects_initializer(
            name, dims, center, tol, bcs_type)

    def add_particle(self, particle):
        self.particles.append(particle)

    def create_part(self, model):

        # create RVE
        sketch = self._create_RVE_sketch(model)

        # create particles geometry in sketch
        for particle in self.particles:
            particle.draw_in_sketch(sketch)

        # create particles parts
        for particle in self.particles:
            particle.create_part(model)

        # create particle instances
        particle_instances = unnest([particle.create_instance(model) for particle in self.particles])
        particle_instances = [instance for instance in particle_instances if instance is not None]
        n_instances = len(particle_instances)

        # create RVE part
        name = '{}_TMP'.format(self.name) if n_instances > 0 else self.name
        tmp_part = self._create_tmp_part(model, sketch, name)

        # create partitions due to particles
        particle_cells = self._create_particles_by_partition(model, tmp_part)

        # create part by merge instances
        if n_instances > 0:
            self._create_part_from_part_particles(model, tmp_part, particle_instances,
                                                  particle_cells)
        else:
            self.info.part = tmp_part
            self._create_instance = True
            region = self.info._get_matrix_regions(tmp_part, particle_cells)
            self._assign_section(region)

        # create required objects (here, because some are required for mesh generation)
        self.obj_creator.create_objs(self.info, model)

    def _create_RVE_sketch(self, model):

        # rectangle points
        pts = []
        for x1, x2 in zip(self.info.bounds[0], self.info.bounds[1]):
            pts.append([x1, x2])

        # sketch
        sketch = model.ConstrainedSketch(name=self.name + '_PROFILE',
                                         sheetSize=2 * self.info.dims[0])
        sketch.rectangle(point1=pts[0], point2=pts[1])

        return sketch

    def _create_particles_by_partition(self, model, tmp_part):

        # create partitions
        particles_by_partition = []
        for particle in self.particles:
            # TODO: how will this comply with 3d?
            success = particle.make_partition(model, tmp_part)
            if success:
                particles_by_partition.append(particle)

        # save regions
        regions = []
        for particle in particles_by_partition:
            regions.append(particle.get_region(tmp_part))

        return unnest(regions)

    def _create_part_from_part_particles(self, model, tmp_part, particle_instances,
                                         particle_regions):

        # initialization
        modelAssembly = model.rootAssembly

        # create initial rbe instance
        rve_tmp_instance = modelAssembly.Instance(name='{}_TMP'.format(self.name),
                                                  part=tmp_part, dependent=ON)

        # cut RVE
        rve_tmp_instance = self._create_part_by_cut(model, rve_tmp_instance,
                                                    particle_instances)

        # assign material
        tmp_rve_part = model.parts[rve_tmp_instance.name[:-2]]
        region = self.info._get_matrix_regions(tmp_rve_part, particle_regions)
        self._assign_section(region, part=tmp_rve_part)

        # merge particles
        rve_tmp_instance = self._create_part_by_merge(model, rve_tmp_instance,
                                                      particle_instances)

        # rename
        if rve_tmp_instance.name != self.name:
            model.parts.changeKey(fromName=rve_tmp_instance.name[:-2],
                                  toName=self.name)
        modelAssembly.features.changeKey(fromName=rve_tmp_instance.name,
                                         toName=self.name)
        self.info.part = model.parts[self.name]

    def _create_part_by_cut(self, model, rve_tmp_instance, particle_instances):

        # initialization
        modelAssembly = model.rootAssembly

        # cut the particles from RVE
        rve_instance = modelAssembly.InstanceFromBooleanCut(
            name='{}_TMP_CUT'.format(self.name),
            instanceToBeCut=rve_tmp_instance,
            cuttingInstances=particle_instances,
            originalInstances=SUPPRESS)
        del modelAssembly.features[rve_tmp_instance.name]

        # resume particles
        for particle_instance in particle_instances:
            modelAssembly.features[particle_instance.name].resume()

        return rve_instance

    def _create_part_by_merge(self, model, rve_tmp_instance, particle_instances):

        # initialization
        modelAssembly = model.rootAssembly

        # get particles with section
        particle_instances_w_sections = []
        for particle_instance in particle_instances:
            if len(particle_instance.part.sectionAssignments) > 0:
                particle_instances_w_sections.append(particle_instance)
            else:
                del modelAssembly.features[particle_instance.name]

        # create merged rve
        if len(particle_instances_w_sections) == 0:
            return rve_tmp_instance

        instances = [rve_tmp_instance] + particle_instances_w_sections
        rve_instance = modelAssembly.InstanceFromBooleanMerge(
            name=self.name, instances=instances, keepIntersections=ON,
            originalInstances=DELETE, domain=GEOMETRY,)

        return rve_instance

    @abstractmethod
    def create_instance(self):
        pass

    def generate_mesh(self, *args, **kwargs):
        # TODO: different mesh in different regions
        return self.mesh.generate_mesh(self.info, *args, **kwargs)

    @property
    def name(self):
        return self.info.name

    @property
    def part(self):
        return self.info.part


class RVE2D(RVE):

    def __init__(self, length, width, center, material, name='RVE', bcs_type='periodic',
                 tol=1e-5):
        dims = (length, width)
        super(RVE2D, self).__init__(name, dims, center, material, tol, bcs_type)

    def _create_tmp_part(self, model, sketch, name):

        # create part
        tmp_part = model.Part(name=name, dimensionality=TWO_D_PLANAR,
                              type=DEFORMABLE_BODY)
        tmp_part.BaseShell(sketch=sketch)

        return tmp_part

    def create_instance(self, model):

        # create instance
        model.rootAssembly.Instance(name=self.name,
                                    part=self.part, dependent=ON)


class RVE3D(RVE):

    def __init__(self, dims, material, name='RVE', center=(0., 0., 0.), tol=1e-5,
                 bcs_type='periodic'):
        super(RVE3D, self).__init__(name, dims, center, material, tol, bcs_type)

    def _create_tmp_part(self, model, sketch, name):

        # create part
        tmp_part = model.Part(name=name, dimensionality=THREE_D,
                              type=DEFORMABLE_BODY)
        tmp_part.BaseSolidExtrude(sketch=sketch, depth=self.info.dims[2])

        return tmp_part

    def create_instance(self, model):

        # verify if already created (e.g. during _create_part_by_merge)
        if not self._create_instance:
            return

        # initialization
        modelAssembly = model.rootAssembly

        # create assembly
        modelAssembly.Instance(name=self.name, part=self.part, dependent=ON)


class RVEInfo(object):

    def __init__(self, name, dims, center, tol):
        self.name = name
        self.dims = dims
        self.center = center
        self.tol = tol
        self.dim = len(self.dims)
        # variable initialization
        self.part = None
        # computed variables
        self.bounds = self._compute_bounds()
        # auxiliar variables
        self.var_coord_map = OrderedDict([('X', 0), ('Y', 1), ('Z', 2)])
        self.sign_bound_map = {'-': 0, '+': 1}

    def _compute_bounds(self):
        bounds = []
        for dim, c in zip(self.dims, self.center):
            bounds.append([c - dim / 2, c + dim / 2])

        return bounds

    def _define_primary_positions(self):
        '''
        Notes
        -----
        Primary positions are edges for 2D and faces for 3d.
        '''
        return [('{}-'.format(var), '{}+'.format(var)) for _, var in zip(self.dims, self.var_coord_map)]

    @ staticmethod
    def _define_positions_by_recursion(ref_positions, d):
        def append_positions(obj, i):
            if i == d:
                positions.append(obj)
            else:
                for ref_obj in (ref_positions[i]):
                    obj_ = obj + ref_obj
                    append_positions(obj_, i + 1)

        positions = []
        obj = ''
        append_positions(obj, 0)

        return positions

    def get_position_from_signs(self, signs, c_vars=None):

        if c_vars is None:
            pos = ''.join(['{}{}'.format(var, sign) for var, sign in zip(self.var_coord_map.keys(), signs)])
        else:
            pos = ''.join(['{}{}'.format(var, sign) for var, sign in zip(c_vars, signs)])

        return pos

    def _get_opposite_position(self, position):
        signs = [sign for sign in position[1::2]]
        opp_signs = self.get_compl_signs(signs)
        c_vars = [var for var in position[::2]]
        return self.get_position_from_signs(opp_signs, c_vars)

    @ staticmethod
    def get_compl_signs(signs):
        c_signs = []
        for sign in signs:
            if sign == '+':
                c_signs.append('-')
            else:
                c_signs.append('+')
        return c_signs

    def _get_edge_sort_direction(self, pos):
        ls = []
        for i in range(0, len(pos), 2):
            ls.append(self.var_coord_map[pos[i]])

        for i in range(3):
            if i not in ls:
                return i

    def get_edge_nodes(self, pos, sort_direction=None, include_vertices=False):

        # get nodes
        edge_name = self.get_edge_name(pos)
        nodes = self.part.sets[edge_name].nodes

        # sort nodes
        if sort_direction is not None:
            nodes = sorted(nodes, key=lambda node: self.get_node_coordinate(node, i=sort_direction))

            # remove vertices
            if not include_vertices:
                nodes = nodes[1:-1]

        # remove vertices if not sorted
        if sort_direction is None and not include_vertices:
            j = self._get_edge_sort_direction(pos)
            x = [self.get_node_coordinate(node, j) for node in nodes]
            f = lambda i: x[i]
            idx_min = min(range(len(x)), key=f)
            idx_max = max(range(len(x)), key=f)
            for index in sorted([idx_min, idx_max], reverse=True):
                del nodes[index]

        return nodes

    @ staticmethod
    def get_node_coordinate(node, i):
        return node.coordinates[i]

    @ staticmethod
    def get_node_coordinate_with_tol(node, i, decimal_places):
        return round(node.coordinates[i], decimal_places)

    @ staticmethod
    def get_edge_name(position):
        return 'EDGE_{}'.format(position)

    @ staticmethod
    def get_vertex_name(position):
        return 'VERTEX_{}'.format(position)

    @ staticmethod
    def get_face_name(position):
        return 'FACE_{}'.format(position)

    def verify_set_name(self, name):
        new_name = name
        i = 1
        while new_name in self.part.sets.keys():
            i += 1
            new_name = '{}_{}'.format(new_name, i)

        return new_name

    @ staticmethod
    def _get_bound_arg_name(pos):
        return '{}Min'.format(pos[0].lower()) if pos[-1] == '+' else '{}Max'.format(pos[0].lower())

    def _get_bound(self, pos):
        i, p = self.var_coord_map[pos[0]], self.sign_bound_map[pos[1]]
        return self.bounds[i][p]

    def _get_matrix_regions(self, part, particle_cells):

        # all part cells
        cells = self._get_cells(part)

        # regions that are not particles
        regions = [cell for cell in cells if cell not in particle_cells]

        return (regions,)


class RVEInfo2D(RVEInfo):

    def __init__(self, name, dims, center, tol):
        super(RVEInfo2D, self).__init__(name, dims, center, tol)
        # additional variables
        self.edge_positions = self._define_primary_positions()
        # define positions
        self.vertex_positions = self._define_positions_by_recursion(
            self.edge_positions, len(self.dims))

    def _get_cells(self, part):
        return part.faces


class RVEInfo3D(RVEInfo):

    def __init__(self, name, dims, center, tol):
        super(RVEInfo3D, self).__init__(name, dims, center, tol)
        # additional variables
        self.face_positions = self._define_primary_positions()
        # define positions
        self.edge_positions = self._define_edge_positions()
        self.vertex_positions = self._define_positions_by_recursion(
            self.face_positions, len(self.dims))

    def _define_edge_positions(self):

        # define positions
        positions = []
        for i, posl in enumerate(self.face_positions[:2]):
            for posr in self.face_positions[i + 1:]:
                for posl_ in posl:
                    for posr_ in posr:
                        positions.append(posl_ + posr_)

        # reagroup positions
        added_positions = []
        edge_positions = []
        for edge in positions:
            if edge in added_positions:
                continue
            opp_edge = self._get_opposite_position(edge)
            edge_positions.append((edge, opp_edge))
            added_positions.extend([edge, opp_edge])

        return edge_positions

    def get_face_nodes(self, face_position, sort_direction_i=None, sort_direction_j=None,
                       include_edges=False):

        # get all nodes
        face_name = self.get_face_name(face_position)
        nodes = list(self.part.sets[face_name].nodes)

        # remove edge nodes
        if not include_edges:
            edge_nodes = self.get_face_edges_nodes(face_position)
            for node in nodes[::-1]:
                if node in edge_nodes:
                    nodes.remove(node)

        # sort nodes
        if sort_direction_i is not None and sort_direction_j is not None:
            d = get_decimal_places(self.tol)
            nodes = sorted(nodes, key=lambda node: (
                self.get_node_coordinate_with_tol(node, i=sort_direction_i, decimal_places=d),
                self.get_node_coordinate_with_tol(node, i=sort_direction_j, decimal_places=d),))

        return MeshNodeArray(nodes)

    def get_face_sort_directions(self, pos):
        k = self.var_coord_map[pos[0]]
        return [i for i in range(3) if i != k]

    def _get_face_edge_positions_names(self, face_position):
        edge_positions = []
        for edge_position in unnest(self.edge_positions):
            if face_position in edge_position:
                edge_positions.append(edge_position)

        return edge_positions

    def get_face_edges_nodes(self, face_position):
        edge_positions = self._get_face_edge_positions_names(face_position)
        edges_nodes = []
        for edge_position in edge_positions:
            edge_name = self.get_edge_name(edge_position)
            edges_nodes.extend(self.part.sets[edge_name].nodes)

        return MeshNodeArray(edges_nodes)

    def get_exterior_edges(self, allow_repetitions=True):

        exterior_edges = []
        for face_position in unnest(self.face_positions):
            kwargs = {self._get_bound_arg_name(face_position): self._get_bound(face_position)}
            edges = self.part.edges.getByBoundingBox(**kwargs)
            exterior_edges.extend(edges)

        # unique edges
        if allow_repetitions:
            return EdgeArray(exterior_edges)
        else:
            edge_indices, unique_exterior_edges = [], []
            for edge in exterior_edges:
                if edge.index not in edge_indices:
                    unique_exterior_edges.append(edge)
                    edge_indices.append(edge.index)

            return EdgeArray(unique_exterior_edges)

    def _get_cells(self, part):
        return part.cells


class RVEInfoPeriodic(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.ref_points_positions = self._define_ref_points()

    def _define_ref_points(self):
        return ['{}-{}+'.format(var, var) for _, var in zip(self.dims, self.var_coord_map)]

    def get_fixed_vertex_position(self):
        return ''.join(['{}-'.format(var) for _, var in zip(self.dims, self.var_coord_map.keys())])

    def get_all_ref_points(self, model=None, only_names=True):
        '''
        Notes
        -----
        Output is given in the order of position definition.
        Model is required if `only_names` is False.
        '''

        if only_names:
            ref_points = [self._get_ref_point_name(position) for position in self.ref_points_positions]
        else:
            ref_points = [model.rootAssembly.sets[self._get_ref_point_name(position)] for position in self.ref_points_positions]

        return ref_points

    @ staticmethod
    def _get_ref_point_name(position):
        return 'REF_POINT_{}'.format(position)


class PeriodicRVEInfo2D(RVEInfo2D, RVEInfoPeriodic):

    def __init__(self, name, dims, center, tol):
        RVEInfo2D.__init__(self, name, dims, center, tol)
        RVEInfoPeriodic.__init__(self)


class PeriodicRVEInfo3D(RVEInfo3D, RVEInfoPeriodic):

    def __init__(self, name, dims, center, tol):
        RVEInfo3D.__init__(self, name, dims, center, tol)
        RVEInfoPeriodic.__init__(self)


class RVEObjCreator(object):

    def create_objs(self, rve_info, *args, **kwargs):
        self._create_bounds_sets(rve_info)

    def _create_bounds_sets(self, rve_info):

        # vertices
        self._create_bound_obj_sets(rve_info, 'vertices',
                                    rve_info.vertex_positions, rve_info.get_vertex_name)

        # edges
        self._create_bound_obj_sets(rve_info, 'edges',
                                    unnest(rve_info.edge_positions), rve_info.get_edge_name)

        # faces
        if rve_info.dim > 2:
            self._create_bound_obj_sets(rve_info, 'faces',
                                        unnest(rve_info.face_positions), rve_info.get_face_name)

    def _create_bound_obj_sets(self, rve_info, obj, positions, get_name):
        '''
        Parameter
        ---------
        obj : str
            Possible values are 'vertices', 'edges', 'faces'
        '''

        # initialization
        get_objs = getattr(rve_info.part, obj)

        # create sets
        for pos in positions:
            name = get_name(pos)
            kwargs = {}
            for i in range(0, len(pos), 2):
                pos_ = pos[i:i + 2]
                var_name = rve_info._get_bound_arg_name(pos_)
                kwargs[var_name] = rve_info._get_bound(pos_)

            objs = get_objs.getByBoundingBox(**kwargs)
            kwargs = {obj: objs}
            rve_info.part.Set(name=name, **kwargs)


class PeriodicRVEObjCreator(RVEObjCreator):

    def create_objs(self, rve_info, model):

        # bound sets
        self._create_bounds_sets(rve_info)

        # reference points
        self._create_ref_points(model, rve_info)

    def _create_ref_points(self, model, rve_info):
        '''
        Notes
        -----
        Any coordinate for reference points position works.
        '''

        # initialization
        modelAssembly = model.rootAssembly
        names = []
        coord = list(rve_info.center)
        if len(coord) == 2:
            coord += [0.]

        # create reference points
        for position in rve_info.ref_points_positions:
            names.append(rve_info._get_ref_point_name(position))
            ref_point = modelAssembly.ReferencePoint(point=coord)
            modelAssembly.Set(name=names[-1],
                              referencePoints=((modelAssembly.referencePoints[ref_point.id],)))

        return names


class BoundaryConditions(object):
    __metaclass__ = ABCMeta

    @ abstractmethod
    def set_bcs(self, *args, **kwargs):
        pass


class PeriodicBoundaryConditions(BoundaryConditions):

    def __init__(self, rve_info):
        super(PeriodicBoundaryConditions, self).__init__()
        self.rve_info = rve_info
        if self.rve_info.dim == 2:
            self.constraints = PBCConstraints2D(rve_info)
        else:
            self.constraints = PBCConstraints3D(rve_info)

    def set_bcs(self, step_name, epsilon, green_lagrange_strain=False):
        '''
        Parameters
        ----------
        epsilon : vector
            Order of elements based on triangular superior strain matrix.
        '''

        # initialization
        disp_bcs = []
        epsilon = symmetricize_vector(epsilon)

        # create strain matrix
        if green_lagrange_strain:
            epsilon = compute_small_strains_from_green(epsilon)

        # fix left bottom node
        position = self.rve_info.get_fixed_vertex_position()
        region_name = '{}.{}'.format(self.rve_info.name, self.rve_info.get_vertex_name(position))
        disp_bcs.append(DisplacementBC(
            name='FIXED_NODE', region=region_name, createStepName=step_name,
            u1=0, u2=0, u3=0))

        # apply displacement
        for k, (position, dim) in enumerate(zip(self.rve_info.ref_points_positions, self.rve_info.dims)):
            region_name = self.rve_info._get_ref_point_name(position)
            applied_disps = {}
            for i in range(len(self.rve_info.dims)):
                applied_disps['u{}'.format(i + 1)] = dim * epsilon[i, k]
            disp_bcs.append(DisplacementBC(
                name='{}'.format(position), region=region_name,
                createStepName=step_name, **applied_disps))

        return self.constraints, disp_bcs


class Constraints(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @ abstractmethod
    def create(self):
        pass


class PBCConstraints(Constraints):

    def __init__(self, rve_info):
        super(PBCConstraints, self).__init__()
        self.rve_info = rve_info

    def create(self, model):

        # get reference points
        ref_points = self.rve_info.get_all_ref_points(only_names=True)

        # apply vertex constraints
        self._apply_vertex_constraints(model, ref_points)

        # apply edge constraints
        self._apply_edge_constraints(model, ref_points)

        # apply face constraints
        if self.rve_info.dim > 2:
            self._apply_face_constraints(model, ref_points)

    def _apply_node_to_node_constraints(self, model, grouped_positions, node_lists,
                                        ref_points_terms_no_dof, constraint_type, dim):

        for i, nodes in enumerate(zip(node_lists[0], node_lists[1])):

            # create set with individual nodes
            set_names = []
            for pos, node in zip(grouped_positions, nodes):
                set_name = '{}_NODE_{}_{}'.format(constraint_type, pos, i)
                self.rve_info.part.Set(name=set_name, nodes=MeshNodeArray((node,)))
                set_names.append('{}.{}'.format(self.rve_info.name, set_name))

            # create constraint
            for ndof in range(1, dim + 1):
                terms = [(1.0, set_names[1], ndof),
                         (-1.0, set_names[0], ndof), ]
                for term in ref_points_terms_no_dof:
                    terms.append(term + [ndof])

                model.Equation(
                    name='{}_CONSTRAINT_{}_{}_{}_{}'.format(
                        constraint_type, grouped_positions[1], grouped_positions[0], i, ndof),
                    terms=terms)

    def _apply_vertex_constraints(self, model, ref_points):
        '''
        Notes
        -----
        The implementation is based on the patterns found on equations (that
        result from the fact that we relate opposite vertices).
        '''

        # local functions
        def get_ref_points_coeff(signs):
            coeffs = []
            for sign in signs:
                if sign == '+':
                    coeffs.append(-1.0)
                else:
                    coeffs.append(1.0)
            return coeffs

        # initialization
        fixed_pos = self.rve_info.get_fixed_vertex_position()

        # apply kinematic constraints
        for k in range((self.rve_info.dim - 1) * 2):
            name = 'VERTEX_CONSTRAINT_'
            signs = ['+' for _ in self.rve_info.dims]
            if k < (self.rve_info.dim - 1) * 2 - 1:  # in one time all the signs are positive
                signs[-(k + 1)] = '-'

            terms_no_dof = []
            for i, signs_ in enumerate([signs, self.rve_info.get_compl_signs(signs)]):
                pos = self.rve_info.get_position_from_signs(signs_)
                if pos != fixed_pos:
                    terms_no_dof.append([(-1.0)**i, '{}.{}'.format(
                        self.rve_info.name, self.rve_info.get_vertex_name(pos)), ])
                    name += pos
            for coeff, ref_point in zip(get_ref_points_coeff(signs), ref_points):
                terms_no_dof.append([coeff, ref_point])

            for ndof in range(1, self.rve_info.dim + 1):
                terms = []
                for term in terms_no_dof:
                    terms.append(term + [ndof])
                model.Equation(name='{}{}'.format(name, ndof),
                               terms=terms)


class PBCConstraints2D(PBCConstraints):

    def __init__(self, rve_info):
        super(PBCConstraints2D, self).__init__(rve_info)

    def _apply_edge_constraints(self, model, ref_points):

        # apply constraints
        for i, (grouped_positions, ref_point) in enumerate(zip(self.rve_info.edge_positions, ref_points)):

            # get sorted nodes
            j = (i + 1) % 2
            node_lists = [self.rve_info.get_edge_nodes(pos, sort_direction=j, include_vertices=False) for pos in grouped_positions]

            # create no_dof terms
            ref_points_terms_no_dof = [[-1.0, ref_point]]

            # create constraints
            self._apply_node_to_node_constraints(
                model, grouped_positions, node_lists, ref_points_terms_no_dof,
                "EDGES", self.rve_info.dim)


class PBCConstraints3D(PBCConstraints):

    def __init__(self, rve_info):
        super(PBCConstraints3D, self).__init__(rve_info)

    def _apply_edge_constraints(self, model, *args, **kwargs):

        for grouped_positions in self.rve_info.edge_positions:

            # get sorted nodes for each edge
            k = self.rve_info._get_edge_sort_direction(grouped_positions[0])
            node_lists = [self.rve_info.get_edge_nodes(pos, sort_direction=k, include_vertices=False) for pos in grouped_positions]

            # create ref_points terms
            ref_point_positions = ['{}-{}+'.format(coord, coord) for coord in grouped_positions[1][::2]]
            sign = -1.0 if grouped_positions[1][-1] == '+' else 1.0
            ref_points_terms_no_dof = [[-1.0, self.rve_info._get_ref_point_name(ref_point_positions[0])],
                                       [sign, self.rve_info._get_ref_point_name(ref_point_positions[1])]]

            # create constraints
            self._apply_node_to_node_constraints(
                model, grouped_positions, node_lists, ref_points_terms_no_dof,
                "EDGES", self.rve_info.dim)

    def _apply_face_constraints(self, model, ref_points):

        # apply constraints
        for ref_point, grouped_positions in zip(ref_points, self.rve_info.face_positions):

            # get nodes
            j, k = self.rve_info.get_face_sort_directions(grouped_positions[0])
            node_lists = [self.rve_info.get_face_nodes(pos, j, k, include_edges=False) for pos in grouped_positions]

            # create no_dof terms
            ref_points_terms_no_dof = [[-1.0, ref_point]]

            # create constraints
            self._apply_node_to_node_constraints(
                model, grouped_positions, node_lists, ref_points_terms_no_dof,
                "FACES", self.rve_info.dim)


class PeriodicMeshGenerator(MeshGenerator):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(PeriodicMeshGenerator, self).__init__()
        self.trial_iter = 1
        self.refine_factor = 1.25

    def generate_mesh(self, rve_info):
        # TODO: generate error file

        it = 1
        success = False
        while it <= self.trial_iter and not success:

            # generate mesh
            self._generate_mesh(rve_info)

            # verify generated mesh
            success = self.mesh_checker.verify_mesh(rve_info)

            # prepare next iteration
            it += 1
            if not success:
                if it <= self.trial_iter:
                    print('Warning: Failed mesh generation. Mesh size will be decreased')
                    self.size /= self.refine_factor
                else:
                    print('Warning: Failed mesh generation')

        return success

    @abstractmethod
    def _generate_mesh(self, rve_info):
        pass


class PeriodicMeshGenerator2D(PeriodicMeshGenerator):

    def __init__(self):
        super(PeriodicMeshGenerator2D, self).__init__()
        self.mesh_checker = PeriodicMeshChecker2D()

    def _generate_mesh(self, rve_info):

        # seed part
        rve_info.part.seedPart(size=self.size,
                               deviationFactor=self.deviation_factor,
                               minSizeFactor=self.min_size_factor)

        # seed edges
        edge_positions = unnest(rve_info.edge_positions)
        edges = [rve_info.part.sets[rve_info.get_edge_name(position)].edges[0] for position in edge_positions]
        rve_info.part.seedEdgeBySize(edges=edges, size=self.size,
                                     deviationFactor=self.deviation_factor,
                                     constraint=FIXED)

        # generate mesh
        rve_info.part.generateMesh()


class PeriodicMeshGenerator3D(PeriodicMeshGenerator):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(PeriodicMeshGenerator3D, self).__init__()
        self.mesh_checker = PeriodicMeshChecker3D()

    def _seed_part(self, rve_info):

        # seed part
        rve_info.part.seedPart(size=self.size,
                               deviationFactor=self.deviation_factor,
                               minSizeFactor=self.min_size_factor)

        # seed exterior edges
        exterior_edges = rve_info.get_exterior_edges()
        rve_info.part.seedEdgeBySize(edges=exterior_edges, size=self.size,
                                     deviationFactor=self.deviation_factor,
                                     constraint=FIXED)

    @abstractmethod
    def _generate_mesh(self, rve_info):
        pass


class PeriodicMeshGenerator3DSimple(PeriodicMeshGenerator3D):

    def _generate_mesh(self, rve_info):

        # delete older mesh
        rve_info.part.deleteMesh()

        # set mesh control
        rve_info.part.setMeshControls(regions=rve_info.part.cells, elemShape=TET,
                                      technique=FREE,)

        # generate mesh
        self._seed_part(rve_info)
        rve_info.part.generateMesh()


class PeriodicMeshGenerator3DS1(PeriodicMeshGenerator3D):

    def _generate_mesh(self, rve_info):

        # delete older mesh
        rve_info.part.deleteMesh()

        # set mesh control
        rve_info.part.setMeshControls(regions=rve_info.part.cells, elemShape=TET,
                                      technique=FREE,)

        # 8-cube partitions (not necessarily midplane)
        planes = [YZPLANE, XZPLANE, XYPLANE]
        for i, plane in enumerate(planes):
            feature = rve_info.part.DatumPlaneByPrincipalPlane(
                principalPlane=plane, offset=rve_info.dims[i] / 2)
            datum = rve_info.part.datums[feature.id]
            rve_info.part.PartitionCellByDatumPlane(datumPlane=datum,
                                                    cells=rve_info.part.cells)

        # reseed (due to partitions)
        self._seed_part(rve_info)

        # z+
        # generate local mesh
        self._mesh_half_periodically(rve_info)
        # transition
        axis = 2
        faces = rve_info.part.faces.getByBoundingBox(zMin=rve_info.bounds[2][1])
        for face in faces:
            self._copy_face_mesh_pattern(rve_info, face, axis)
        # z-
        self._mesh_half_periodically(rve_info, zp=False)

    def _mesh_half_periodically(self, rve_info, zp=True):

        def get_mean_coord(axis):
            return (rve_info.bounds[axis][0] + rve_info.bounds[axis][1]) / 2.

        # initialization
        kwarg_name = 'zMin' if zp else 'zMax'
        kwargs = {kwarg_name: get_mean_coord(2)}

        # generate mesh in picked cells
        picked_cells = rve_info.part.cells.getByBoundingBox(
            xMin=get_mean_coord(0), yMin=get_mean_coord(1), **kwargs)
        rve_info.part.generateMesh(regions=picked_cells)

        # copy pattern and generate mesh in selected cells
        axes = (0, 1, 0)
        face_bounds = ({'xMin': rve_info.bounds[0][1], 'yMin': get_mean_coord(1)},
                       {'yMin': rve_info.bounds[1][1]},
                       {'xMax': rve_info.bounds[0][0], 'yMax': get_mean_coord(1)},)
        cell_bounds = ({'xMax': get_mean_coord(0), 'yMin': get_mean_coord(1)},
                       {'xMax': get_mean_coord(0), 'yMax': get_mean_coord(1)},
                       {'xMin': get_mean_coord(0), 'yMax': get_mean_coord(1)})

        for axis, face_bounds_, cell_bounds_ in zip(axes, face_bounds, cell_bounds):
            # copy pattern
            face_bounds_.update(kwargs)
            faces = rve_info.part.faces.getByBoundingBox(**face_bounds_)
            for face in faces:
                self._copy_face_mesh_pattern(rve_info, face, axis)

            # generate local mesh
            cell_bounds_.update(kwargs)
            picked_cells = rve_info.part.cells.getByBoundingBox(**cell_bounds_)
            rve_info.part.generateMesh(regions=picked_cells)

    def _copy_face_mesh_pattern(self, rve_info, face, axis):

        # find k
        tf_names = [name for name in rve_info.part.sets.keys() if '_TARGET_FACE_' in name]
        k = int(tf_names[-1].split('_')[-1]) + 1 if len(tf_names) > 0 else 0

        # get opposite edge point
        pt = list(face.pointOn[0])
        to = 0 if abs(pt[axis] - rve_info.bounds[axis][1]) < rve_info.tol else 1
        pt[axis] = rve_info.bounds[axis][to]

        # find opposite face and create sets
        target_face = rve_info.part.faces.findAt(pt)
        face_set = rve_info.part.Set(name='_FACE_{}'.format(k), faces=FaceArray((face,)))
        rve_info.part.Set(name='_TARGET_FACE_{}'.format(k), faces=FaceArray((target_face,)))
        vertex_indices = face.getVertices()
        vertices = [rve_info.part.vertices[index].pointOn[0] for index in vertex_indices]
        coords = [list(vertex) for vertex in vertices]
        for coord in coords:
            coord[axis] = rve_info.bounds[axis][to]
        nodes = [face_set.nodes.getClosest(vertex) for vertex in vertices]
        rve_info.part.copyMeshPattern(faces=face_set, targetFace=target_face,
                                      nodes=nodes,
                                      coordinates=coords)


class PeriodicMeshChecker(object):

    def __init__(self):
        pass

    def _verify_edges(self, rve_info):

        for grouped_positions in rve_info.edge_positions:

            # get sorted nodes for each edge
            k = rve_info._get_edge_sort_direction(grouped_positions[0])
            node_lists = [rve_info.get_edge_nodes(pos, sort_direction=k, include_vertices=False) for pos in grouped_positions]

            # verify sizes
            sizes = [len(node_list) for node_list in node_lists]
            if len(set(sizes)) > 1:
                return False

            # verify if tolerance is respected
            for node_list in node_lists[1:]:
                for n, node in enumerate(node_lists[0]):
                    if not self._verify_tol_edge_nodes(rve_info, node, node_list[n], k):
                        return False

        return True

    def _verify_tol_edge_nodes(self, rve_info, node, node_cmp, k):
        if abs(node.coordinates[k] - node_cmp.coordinates[k]) > rve_info.tol:
            # create set with error nodes
            set_name = rve_info.verify_set_name('_ERROR_EDGE_NODES')
            rve_info.part.Set(set_name, nodes=MeshNodeArray(node, node_cmp))
            return False

        return True


class PeriodicMeshChecker2D(PeriodicMeshChecker):

    def __init__(self):
        super(PeriodicMeshChecker2D, self).__init__()

    def verify_mesh(self, rve_info):
        '''
        Verify correctness of generated mesh based on allowed tolerance. It
        immediately stops when a node pair does not respect the tolerance.
        '''
        return self._verify_edges(rve_info)


class PeriodicMeshChecker3D(PeriodicMeshChecker):
    # TODO: consider to have only by sorting due to way constraints are generated
    # TODO: consider to split class into 2 -> in practice it is better to use strategy pattern

    def __init__(self):
        super(PeriodicMeshChecker3D, self).__init__()
        self.face_by_closest = False

    def verify_mesh(self, rve_info):
        '''
        Verify correctness of generated mesh based on allowed tolerance. It
        immediately stops when a node pair does not respect the tolerance.
        '''
        # TODO: create get all error nodes

        # verify edges
        if not self._verify_edges(rve_info):
            return False

        # verify faces
        if self.face_by_closest:
            return self._verify_faces_by_closest(rve_info)
        else:
            return self._verify_faces_by_sorting(rve_info)

    def _verify_faces_by_sorting(self, rve_info):
        '''
        Notes
        -----
        1. the sort method is less robust due to rounding errors. `mesh_tol`
        is used to increase its robustness, but find by closest should be
        preferred.
        '''

        for grouped_positions in rve_info.face_positions:

            # get nodes
            pos_i, pos_j = grouped_positions
            j, k = rve_info.get_face_sort_directions(pos_i)
            nodes_i = rve_info.get_face_nodes(pos_i, j, k)
            nodes_j = rve_info.get_face_nodes(pos_j, j, k)

            # verify size
            if len(nodes_i) != len(nodes_j):
                return False

            # verify if tolerance is respected
            for n, (node, node_cmp) in enumerate(zip(nodes_i, nodes_j)):

                # verify tolerance
                if not self._verify_tol_face_nodes(rve_info, node, node_cmp, j, k):
                    return False

        return True

    def _verify_faces_by_closest(self, rve_info):

        for grouped_positions in rve_info.face_positions:

            # get nodes
            pos_i, pos_j = grouped_positions
            j, k = rve_info.get_face_sort_directions(pos_i)
            nodes_i = rve_info.get_face_nodes(pos_i)
            nodes_j = rve_info.get_face_nodes(pos_j)

            # verify size
            if len(nodes_i) != len(nodes_j):
                return False

            # verify if tolerance is respected
            for node in nodes_i:
                node_cmp = nodes_j.getClosest(node.coordinates)

                # verify tolerance
                if not self._verify_tol_face_nodes(rve_info, node, node_cmp, j, k):
                    return False

        return True

    def _verify_tol_face_nodes(self, rve_info, node, node_cmp, j, k):
        if abs(node.coordinates[j] - node_cmp.coordinates[j]) > rve_info.tol or abs(node.coordinates[k] - node_cmp.coordinates[k]) > rve_info.tol:
            # create set with error nodes
            set_name = rve_info.verify_set_name('_ERROR_FACE_NODES')
            rve_info.part.Set(set_name, nodes=MeshNodeArray((node, node_cmp)))
            return False

        return True
