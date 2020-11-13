'''
Created on 2020-03-24 14:33:48
Last modified on 2020-11-13 11:33:37

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create an RVE class from which more useful classes can inherit.

Notes
-----
-Based on code developed by M. A. Bessa.

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
                             YZPLANE, XZPLANE, XYPLANE)
from part import (EdgeArray, FaceArray)
from mesh import MeshNodeArray

# standard
from abc import abstractmethod
from abc import ABCMeta
from collections import OrderedDict

# local library
from ..modelling.bcs import DisplacementBC
from ...utils.linalg import symmetricize_vector
from ...utils.solid_mechanics import compute_small_strains_from_green


# TODO: create a Geometry abstract class
# TODO: handle warnings and errors using particular class
# TODO: base default mesh size in characteristic length


# object definition

class RVE(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, dims, center):
        self.name = name
        self.dims = dims
        self.center = center
        # variable initialization
        self.part = None
        self.particles = []
        # computed variables
        self.bounds = self._compute_bounds()
        # auxiliar variables
        self.var_coord_map = OrderedDict([('X', 0), ('Y', 1), ('Z', 2)])
        self.sign_bound_map = {'-': 0, '+': 1}
        # additional variables
        self.ref_points_positions = self._define_ref_points()
        # initial operations
        self._compute_bounds()

    def add_particle(self, particle):
        self.particles.append(particle)

    def apply_bcs_disp(self, model, step_name, epsilon,
                       green_lagrange_strain=False):
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
        position = self._get_fixed_vertex_position()
        region_name = '{}.{}'.format(self.name, self._get_vertex_name(position))
        disp_bcs.append(DisplacementBC(
            name='FIXED_NODE', region=region_name, createStepName=step_name,
            u1=0, u2=0, u3=0))

        # apply displacement
        for k, (position, dim) in enumerate(zip(self.ref_points_positions, self.dims)):
            region_name = self._get_ref_point_name(position)
            applied_disps = {}
            for i in range(len(self.dims)):
                applied_disps['u{}'.format(i + 1)] = dim * epsilon[i, k]
            disp_bcs.append(DisplacementBC(
                name='{}'.format(position), region=region_name,
                createStepName=step_name, **applied_disps))

        # TODO: return disp bcs and do apply outside?
        for disp_bc in disp_bcs:
            disp_bc.apply_bc(model)

    def _create_RVE_sketch(self, model):

        # rectangle points
        pts = []
        for x1, x2 in zip(self.bounds[0], self.bounds[1]):
            pts.append([x1, x2])

        # sketch
        sketch = model.ConstrainedSketch(name=self.name + '_PROFILE',
                                         sheetSize=2 * self.dims[0])
        sketch.rectangle(point1=pts[0], point2=pts[1])

        return sketch


class RVE2D(RVE):

    def __init__(self, length, width, center, name='RVE'):
        super(RVE2D, self).__init__(name, dims=(length, width), center=center)
        # additional variables
        self.edge_positions = self._define_primary_positions()
        # define positions
        self.vertex_positions = self._define_positions_by_recursion(
            self.edge_positions, len(self.dims))

    def create_part(self, model):
        # TODO: move to parent?

        # create RVE
        sketch = self._create_RVE_sketch(model)

        # create particles geometry in sketch
        for particle in self.particles:
            particle.create_inner_geometry(sketch, self)

        # create part
        self.part = model.Part(name=self.name, dimensionality=TWO_D_PLANAR,
                               type=DEFORMABLE_BODY)
        self.part.BaseShell(sketch=sketch)

        # TODO: create particles by partitions

        # create PBCs sets (here because sets are required for meshing purposes)
        self._create_bounds_sets()

    def create_instance(self, model):

        # create instance
        model.rootAssembly.Instance(name=self.name,
                                    part=self.part, dependent=ON)

    def generate_mesh(self):

        # seed part
        self.part.seedPart(size=self.mesh_size,
                           deviationFactor=self.mesh_deviation_factor,
                           minSizeFactor=self.mesh_min_size_factor)

        # seed edges
        edge_positions = self._unnest(self.edge_positions)
        edges = [self.part.sets[self._get_edge_name(position)].edges[0] for position in edge_positions]
        self.part.seedEdgeBySize(edges=edges, size=self.mesh_size,
                                 deviationFactor=self.mesh_deviation_factor,
                                 constraint=FIXED)
        # generate mesh
        self.part.generateMesh()


class RVE3D(RVE):

    def __init__(self, dims, name='RVE', center=(0., 0., 0.)):
        super(RVE3D, self).__init__(name, dims, center)
        # additional variables
        self.face_positions = self._define_primary_positions()
        # define positions
        self.edge_positions = self._define_edge_positions()
        self.vertex_positions = self._define_positions_by_recursion(
            self.face_positions, len(self.dims))

    def create_part(self, model):

        # create RVE
        sketch = self._create_RVE_sketch(model)

        # create particles geometry in sketch
        for particle in self.particles:
            particle.create_inner_geometry(sketch, self)

        # create particles parts
        for particle in self.particles:
            particle.create_part(model, self)

        # create part
        part_name = '{}_TMP'.format(self.name) if len(model.parts) > 0 else self.name
        self.part = model.Part(name=part_name, dimensionality=THREE_D,
                               type=DEFORMABLE_BODY)
        self.part.BaseSolidExtrude(sketch=sketch, depth=self.dims[2])

        # create part by merge instances
        if len(model.parts) > 1:
            self._create_part_by_merge(model)

        # create PBCs sets (here because sets are required for meshing purposes)
        self._create_bounds_sets()

    def _create_part_by_merge(self, model):
        # TODO: make with voids to extend method (base it on the material); make also combined

        # initialization
        modelAssembly = model.rootAssembly

        # create rve instance
        modelAssembly.Instance(name='{}_TMP'.format(self.name),
                               part=self.part, dependent=ON)

        # create particle instances
        for particle in self.particles:
            particle.create_instance(model)

        # create merged rve
        modelAssembly.InstanceFromBooleanMerge(name=self.name,
                                               instances=modelAssembly.instances.values(),
                                               keepIntersections=ON,
                                               originalInstances=DELETE,
                                               domain=GEOMETRY)
        modelAssembly.features.changeKey(fromName='{}-1'.format(self.name),
                                         toName='RVE')
        self.part = model.parts[self.name]  # override part

    def create_instance(self, model):

        # initialization
        modelAssembly = model.rootAssembly

        # verify if already created (e.g. during _create_part_by_merge)
        if len(modelAssembly.instances.keys()) > 0:
            return

        # create assembly
        modelAssembly.Instance(name=self.name, part=self.part, dependent=ON)

    def generate_mesh(self, face_by_closest=True, simple_trial=False):

        # set mesh control
        self.part.setMeshControls(regions=self.part.cells, elemShape=TET,
                                  technique=FREE,)

        # generate mesh by simple strategy
        if simple_trial:
            # generate mesh
            self._seed_part()
            self.part.generateMesh()

            # verify mesh
            success = self.verify_mesh_for_pbcs(face_by_closest=face_by_closest)

        # TODO: delete
        print('Mesh correcly generated? {}'.format(success))
        return success

        # retry meshing if unsuccessful
        if not simple_trial or not success:
            if simple_trial:
                print("Warning: Unsucessful mesh generation. Another strategy will be tried out")

            # retry meshing
            self._retry_meshing()

            # verify mesh
            success = self.verify_mesh_for_pbcs(face_by_closest=face_by_closest)

            if not success:
                print("Warning: Unsucessful mesh generation")


class RVEInfo(object):
    # TODO: split between container and set creator?

    def __init__(self):
        pass

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

    def _define_ref_points(self):
        return ['{}-{}+'.format(var, var) for _, var in zip(self.dims, self.var_coord_map)]

    def _get_fixed_vertex_position(self):
        return ''.join(['{}-'.format(var) for _, var in zip(self.dims, self.var_coord_map.keys())])

    def _get_position_from_signs(self, signs, c_vars=None):

        if c_vars is None:
            pos = ''.join(['{}{}'.format(var, sign) for var, sign in zip(self.var_coord_map.keys(), signs)])
        else:
            pos = ''.join(['{}{}'.format(var, sign) for var, sign in zip(c_vars, signs)])

        return pos

    def _get_opposite_position(self, position):
        signs = [sign for sign in position[1::2]]
        opp_signs = self._get_compl_signs(signs)
        c_vars = [var for var in position[::2]]
        return self._get_position_from_signs(opp_signs, c_vars)

    @staticmethod
    def _get_compl_signs(signs):
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

    def _get_edge_nodes(self, pos, sort_direction=None, include_vertices=False):

        # get nodes
        edge_name = self._get_edge_name(pos)
        nodes = self.part.sets[edge_name].nodes

        # sort nodes
        if sort_direction is not None:
            nodes = sorted(nodes, key=lambda node: self._get_node_coordinate(node, i=sort_direction))

            # remove vertices
            if not include_vertices:
                nodes = nodes[1:-1]

        # remove vertices if not sorted
        if sort_direction is None and not include_vertices:
            j = self._get_edge_sort_direction(pos)
            x = [self._get_node_coordinate(node, j) for node in nodes]
            f = lambda i: x[i]
            idx_min = min(range(len(x)), key=f)
            idx_max = max(range(len(x)), key=f)
            for index in sorted([idx_min, idx_max], reverse=True):
                del nodes[index]

        return nodes

    @staticmethod
    def _get_node_coordinate(node, i):
        return node.coordinates[i]

    @staticmethod
    def _get_node_coordinate_with_tol(node, i, decimal_places):
        return round(node.coordinates[i], decimal_places)

    def _get_decimal_places(self):
        d = 0
        aux = 1
        while aux > self.mesh_tol:
            d += 1
            aux = 10**(-d)

        return d

    @staticmethod
    def _get_edge_name(position):
        return 'EDGE_{}'.format(position)

    @staticmethod
    def _get_vertex_name(position):
        return 'VERTEX_{}'.format(position)

    @staticmethod
    def _get_face_name(position):
        return 'FACE_{}'.format(position)

    @staticmethod
    def _get_ref_point_name(position):
        return 'REF_POINT_{}'.format(position)

    def _get_all_ref_points(self, model=None, only_names=False):
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

    def _verify_set_name(self, name):
        new_name = name
        i = 1
        while new_name in self.part.sets.keys():
            i += 1
            new_name = '{}_{}'.format(new_name, i)

        return new_name

    @staticmethod
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

    @staticmethod
    def _get_bound_arg_name(pos):
        return '{}Min'.format(pos[0].lower()) if pos[-1] == '+' else '{}Max'.format(pos[0].lower())

    def _get_bound(self, pos):
        i, p = self.var_coord_map[pos[0]], self.sign_bound_map[pos[1]]
        return self.bounds[i][p]

class RVEInfo2D(RVEInfo):

    def __init__(self):
        super(RVEInfo2D, self).__init__()


class RVEInfo3D(RVEInfo):

    def __init__(self):
        super(RVEInfo3D, self).__init__()

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
            added_positions.append(edge)

        return edge_positions

    def _get_face_nodes(self, face_position, sort_direction_i=None, sort_direction_j=None,
                        include_edges=False):

        # get all nodes
        face_name = self._get_face_name(face_position)
        nodes = list(self.part.sets[face_name].nodes)

        # remove edge nodes
        if not include_edges:
            edge_nodes = self._get_face_edges_nodes(face_position)
            for node in nodes[::-1]:
                if node in edge_nodes:
                    nodes.remove(node)

        # sort nodes
        if sort_direction_i is not None and sort_direction_j is not None:
            d = self._get_decimal_places()
            nodes = sorted(nodes, key=lambda node: (
                self._get_node_coordinate_with_tol(node, i=sort_direction_i, decimal_places=d),
                self._get_node_coordinate_with_tol(node, i=sort_direction_j, decimal_places=d),))

        return nodes

    def _get_face_sort_directions(self, pos):
        k = self.var_coord_map[pos[0]]
        return [i for i in range(3) if i != k]

    def _get_face_edge_positions_names(self, face_position):
        edge_positions = []
        for edge_position in self._unnest(self.edge_positions):
            if face_position in edge_position:
                edge_positions.append(edge_position)

        return edge_positions

    def _get_face_edges_nodes(self, face_position):
        edge_positions = self._get_face_edge_positions_names(face_position)
        edges_nodes = []
        for edge_position in edge_positions:
            edge_name = self._get_edge_name(edge_position)
            edges_nodes.extend(self.part.sets[edge_name].nodes)

        return edges_nodes

    def _get_exterior_edges(self, allow_repetitions=True):

        exterior_edges = []
        for face_position in self._unnest(self.face_positions):
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


class RVEObjCreation(object):

    def __init__(self):
        pass

    def _create_bounds_sets(self):

        # vertices
        self._create_bound_obj_sets('vertices', self.vertex_positions, self._get_vertex_name)

        # edges
        self._create_bound_obj_sets('edges', self._unnest(self.edge_positions), self._get_edge_name)

        # faces
        if len(self.dims) > 2:
            self._create_bound_obj_sets('faces', self._unnest(self.face_positions), self._get_face_name)

    def _create_bound_obj_sets(self, obj, positions, get_name):
        '''
        Parameter
        ---------
        obj : str
            Possible values are 'vertices', 'edges', 'faces'
        '''

        # initialization
        get_objs = getattr(self.part, obj)

        # create sets
        for pos in positions:
            name = get_name(pos)
            kwargs = {}
            for i in range(0, len(pos), 2):
                pos_ = pos[i:i + 2]
                var_name = self._get_bound_arg_name(pos_)
                kwargs[var_name] = self._get_bound(pos_)

            objs = get_objs.getByBoundingBox(**kwargs)
            kwargs = {obj: objs}
            self.part.Set(name=name, **kwargs)

    def _create_pbcs_ref_points(self, model):
        '''
        Notes
        -----
        Any coordinate for reference points position works.
        '''

        # initialization
        modelAssembly = model.rootAssembly
        names = []
        coord = list(self.center)
        if len(coord) == 2:
            coord += [0.]

        # create reference points
        for position in self.ref_points_positions:
            names.append(self._get_ref_point_name(position))
            ref_point = modelAssembly.ReferencePoint(point=coord)
            modelAssembly.Set(name=names[-1],
                              referencePoints=((modelAssembly.referencePoints[ref_point.id],)))

        return names


class Constraints(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def apply_constraints(self):
        pass


class PBCConstraints(Constraints):

    def __init__(self):
        super(self, PBCConstraints).__init__()

    def apply_constraints(self, model):

        # initialization
        dim = len(self.dims)

        # create reference points
        ref_points = self._create_pbcs_ref_points(model)

        # apply vertex constraints
        self._apply_vertex_constraints(model, dim, ref_points)

        # apply edge constraints
        self._apply_edge_constraints(model, dim, ref_points)

        # apply face constraints
        if dim > 2:
            self._apply_face_constraints(model, dim, ref_points)

    def _apply_node_to_node_constraint(self, model, grouped_positions, node_lists,
                                       ref_points_terms_no_dof, constraint_type, dim):

        for i, nodes in enumerate(zip(node_lists[0], node_lists[1])):

            # create set with individual nodes
            set_names = []
            for pos, node in zip(grouped_positions, nodes):
                set_name = '{}_NODE_{}_{}'.format(constraint_type, pos, i)
                self.part.Set(name=set_name, nodes=MeshNodeArray((node,)))
                set_names.append('{}.{}'.format(self.name, set_name))

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


class PBCConstraints2D(PBCConstraints):

    def __init__(self):
        super(PBCConstraints, self).__init__()

    def _apply_edge_constraints(self, model, dim, ref_points):

        # apply constraints
        for i, (grouped_positions, ref_point) in enumerate(zip(self.edge_positions, ref_points)):

            # get sorted nodes
            j = (i + 1) % 2
            node_lists = [self._get_edge_nodes(pos, sort_direction=j, include_vertices=False) for pos in grouped_positions]

            # create no_dof terms
            ref_points_terms_no_dof = [[-1.0, ref_point]]

            # create constraints
            self._apply_node_to_node_constraints(
                model, grouped_positions, node_lists, ref_points_terms_no_dof,
                "EDGES", dim)

    def _apply_vertex_constraints(self, model, dim, ref_points):
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
        fixed_pos = self._get_fixed_vertex_position()

        # apply kinematic constraints
        for k in range((dim - 1) * 2):
            name = 'VERTEX_CONSTRAINT_'
            signs = ['+' for _ in self.dims]
            if k < (dim - 1) * 2 - 1:  # in one time all the signs are positive
                signs[-(k + 1)] = '-'

            terms_no_dof = []
            for i, signs_ in enumerate([signs, self._get_compl_signs(signs)]):
                pos = self._get_position_from_signs(signs_)
                if pos != fixed_pos:
                    terms_no_dof.append([(-1.0)**i, '{}.{}'.format(self.name, self._get_vertex_name(pos)), ])
                    name += pos
            for coeff, ref_point in zip(get_ref_points_coeff(signs), ref_points):
                terms_no_dof.append([coeff, ref_point])

            for ndof in range(1, dim + 1):
                terms = []
                for term in terms_no_dof:
                    terms.append(term + [ndof])
                model.Equation(name='{}{}'.format(name, ndof),
                               terms=terms)


class PBCConstraints3D(PBCConstraints):

    def __init__(self):
        super(PBCConstraints, self).__init__()

    def _apply_edge_constraints(self, model, dim, *args):

        for grouped_positions in self.edge_positions:

            # get sorted nodes for each edge
            k = self._get_edge_sort_direction(grouped_positions[0])
            node_lists = [self._get_edge_nodes(pos, sort_direction=k, include_vertices=False) for pos in grouped_positions]

            # create ref_points terms
            ref_point_positions = ['{}-{}+'.format(coord, coord) for coord in grouped_positions[1][::2]]
            sign = -1.0 if grouped_positions[1][-1] == '+' else 1.0
            ref_points_terms_no_dof = [[-1.0, self._get_ref_point_name(ref_point_positions[0])],
                                       [sign, self._get_ref_point_name(ref_point_positions[1])]]

            # create constraints
            self._apply_node_to_node_constraints(
                model, grouped_positions, node_lists, ref_points_terms_no_dof,
                "EDGES", dim)

    def _apply_face_constraints(self, model, dim, ref_points):

        # apply constraints
        for ref_point, grouped_positions in zip(ref_points, self.face_positions):

            # get nodes
            j, k = self._get_face_sort_directions(grouped_positions[0])
            node_lists = [self._get_face_nodes(pos, j, k, include_edges=False) for pos in grouped_positions]

            # create no_dof terms
            ref_points_terms_no_dof = [[-1.0, ref_point]]

            # create constraints
            self._apply_node_to_node_constraints(
                model, grouped_positions, node_lists, ref_points_terms_no_dof,
                "FACES", dim)


class MeshGenerator(object):

    def __init__(self):
        self.tol = 1e-5
        self.size = .02
        self.deviation_factor = .4
        self.min_size_factor = .4

    def change_mesh_definitions(self, **kwargs):
        '''
        See mesh definition at __init__ to find out the variables that can be
        changed.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)


class PeriodicMeshGenerator(object):

    def __init__(self):
        super(PeriodicMeshGenerator, self).__init__()
        self.trial_iter = 1
        self.refine_factor = 1.25


class PeriodicMeshGenerator2D(PeriodicMeshGenerator):

    def __init__(self):
        super(PeriodicMeshGenerator, self).__init__()

    def generate_mesh_for_pbcs(self):
        # TODO: generate error file

        it = 1
        success = False
        while it <= self.mesh_trial_iter and not success:

            # generate mesh
            self.generate_mesh()

            # verify generated mesh
            success = self.verify_mesh_for_pbcs()

            # prepare next iteration
            it += 1
            if not success:
                if it <= self.mesh_trial_iter:
                    print('Warning: Failed mesh generation. Mesh size will be decreased')
                    self.mesh_size /= self.mesh_refine_factor
                else:
                    print('Warning: Failed mesh generation')

        return success


class PeriodicMeshGenerator3D(PeriodicMeshGenerator):

    def __init__(self):
        super(PeriodicMeshGenerator, self).__init__()

    def _seed_part(self):
        # seed part
        self.part.seedPart(size=self.mesh_size,
                           deviationFactor=self.mesh_deviation_factor,
                           minSizeFactor=self.mesh_min_size_factor)

        # seed exterior edges
        exterior_edges = self._get_exterior_edges()
        self.part.seedEdgeBySize(edges=exterior_edges, size=self.mesh_size,
                                 deviationFactor=self.mesh_deviation_factor,
                                 constraint=FIXED)

    def _mesh_half_periodically(self, k=0, zp=True):
        # TODO: need to be tested with the use of bounds

        # initialization
        if zp:
            kwargs = {'zMin': self.dims[2] / 2}
        else:
            kwargs = {'zMax': self.dims[2] / 2}

        pickedCells = self.part.cells.getByBoundingBox(xMin=self.dims[0] / 2,
                                                       yMin=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)
        # copy pattern
        axis = 0
        faces = self.part.faces.getByBoundingBox(xMin=self.dims[0],
                                                 yMin=self.dims[1] / 2,
                                                 **kwargs)
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k)
            k += 1
        # generate local mesh
        pickedCells = self.part.cells.getByBoundingBox(xMax=self.dims[0] / 2,
                                                       yMin=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)
        # copy pattern
        axis = 1
        faces = self.part.faces.getByBoundingBox(yMin=self.dims[1],
                                                 **kwargs)
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k)
            k += 1
        # generate local mesh
        pickedCells = self.part.cells.getByBoundingBox(xMax=self.dims[0] / 2,
                                                       yMax=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)
        # copy pattern
        axis = 0
        faces = self.part.faces.getByBoundingBox(xMax=0.,
                                                 yMax=self.dims[1] / 2,
                                                 **kwargs)
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k, s=1)
            k += 1
        # generate local mesh
        pickedCells = self.part.cells.getByBoundingBox(xMin=self.dims[0] / 2,
                                                       yMax=self.dims[1] / 2,
                                                       **kwargs)
        self.part.generateMesh(regions=pickedCells)

        return k

    def _copy_face_mesh_pattern(self, face, axis, k, s=0):
        pt = list(face.pointOn[0])
        pt[axis] = self.dims[axis] * s
        target_face = self.part.faces.findAt(pt)
        face_set = self.part.Set(name='_FACE_{}'.format(k), faces=FaceArray((face,)))
        self.part.Set(name='_TARGET_FACE_{}'.format(k), faces=FaceArray((target_face,)))
        k += 1
        vertex_indices = face.getVertices()
        vertices = [self.part.vertices[index].pointOn[0] for index in vertex_indices]
        coords = [list(vertex) for vertex in vertices]
        for coord in coords:
            coord[axis] = self.dims[axis] * s
        nodes = [face_set.nodes.getClosest(vertex) for vertex in vertices]
        self.part.copyMeshPattern(faces=face_set, targetFace=target_face,
                                  nodes=nodes,
                                  coordinates=coords)

    def _retry_meshing(self):

        # delete older mesh
        self.part.deleteMesh()

        # 8-cube partitions (not necessarily midplane)
        planes = [YZPLANE, XZPLANE, XYPLANE]
        for i, plane in enumerate(planes):
            feature = self.part.DatumPlaneByPrincipalPlane(principalPlane=plane,
                                                           offset=self.dims[i] / 2)
            datum = self.part.datums[feature.id]
            self.part.PartitionCellByDatumPlane(datumPlane=datum, cells=self.part.cells)

        # reseed (due to partitions)
        self._seed_part()

        # z+
        # generate local mesh
        k = self._mesh_half_periodically()
        # transition
        axis = 2
        faces = self.part.faces.getByBoundingBox(zMin=self.dims[2])
        for face in faces:
            self._copy_face_mesh_pattern(face, axis, k)
            k += 1
        # z-
        self._mesh_half_periodically(k, zp=False)


class MeshChecker(object):

    def __init__(self):
        pass

    def _verify_edges(self):

        for grouped_positions in self.edge_positions:

            # get sorted nodes for each edge
            k = self._get_edge_sort_direction(grouped_positions[0])
            node_lists = [self._get_edge_nodes(pos, sort_direction=k, include_vertices=False) for pos in grouped_positions]

            # verify sizes
            sizes = [len(node_list) for node_list in node_lists]
            if len(set(sizes)) > 1:
                return False

            # verify if tolerance is respected
            for node_list in node_lists[1:]:
                for n, node in enumerate(node_lists[0]):
                    if not self._verify_tol_edge_nodes(node, node_list[n], k):
                        return False

        return True

    def _verify_tol_edge_nodes(self, node, node_cmp, k):
        if abs(node.coordinates[k] - node_cmp.coordinates[k]) > self.mesh_tol:
            # create set with error nodes
            set_name = self._verify_set_name('_ERROR_EDGE_NODES')
            self.part.Set(set_name, nodes=MeshNodeArray(node, node_cmp))
            return False

        return True


class MeshChecker2D(MeshChecker):

    def __init__(self):
        super(MeshChecker2D, self).__init__()

    def verify_mesh_for_pbcs(self):
        '''
        Verify correctness of generated mesh based on allowed tolerance. It
        immediately stops when a node pair does not respect the tolerance.
        '''
        return self._verify_edges()


class MeshChecker3D(MeshChecker):

    def __init__(self):
        super(MeshChecker3D, self).__init__()

    def verify_mesh_for_pbcs(self, face_by_closest=True):
        '''
        Verify correctness of generated mesh based on allowed tolerance. It
        immediately stops when a node pair does not respect the tolerance.
        '''
        # TODO: create get all error nodes

        # verify edges
        if not self._verify_edges():
            return False

        # verify faces
        if face_by_closest:
            return self._verify_faces_by_closest()
        else:
            return self._verify_faces_by_sorting()

    def _verify_faces_by_sorting(self):
        '''
        Notes
        -----
        1. the sort method is less robust due to rounding errors. `mesh_tol`
        is used to increase its robustness, but find by closest should be
        preferred.
        '''

        for grouped_positions in self.face_positions:

            # get nodes
            pos_i, pos_j = grouped_positions
            j, k = self._get_face_sort_directions(pos_i)
            nodes_i = self._get_face_nodes(pos_i, j, k)
            nodes_j = self._get_face_nodes(pos_j, j, k)

            # verify size
            if len(nodes_i) != len(nodes_j):
                return False

            # verify if tolerance is respected
            for n, (node, node_cmp) in enumerate(zip(nodes_i, nodes_j)):

                # verify tolerance
                if not self._verify_tol_face_nodes(node, node_cmp, j, k):
                    return False

        return True

    def _verify_faces_by_closest(self):

        for grouped_positions in self.face_positions:

            # get nodes
            pos_i, pos_j = grouped_positions
            j, k = self._get_face_sort_directions(pos_i)
            nodes_i = self._get_face_nodes(pos_i)
            nodes_j = self._get_face_nodes(pos_j)

            # verify size
            if len(nodes_i) != len(nodes_j):
                return False

            # verify if tolerance is respected
            for node in nodes_i:
                node_cmp = nodes_j.getClosest(node.coordinates)

                # verify tolerance
                if not self._verify_tol_face_nodes(node, node_cmp, j, k):
                    return False

        return True

    def _verify_tol_face_nodes(self, node, node_cmp, j, k):
        if abs(node.coordinates[j] - node_cmp.coordinates[j]) > self.mesh_tol or abs(node.coordinates[k] - node_cmp.coordinates[k]) > self.mesh_tol:
            # create set with error nodes
            set_name = self._verify_set_name('_ERROR_FACE_NODES')
            self.part.Set(set_name,
                          nodes=MeshNodeArray((node, node_cmp)))
            return False

        return True


def _unnest(array):
    # TODO: unnest more levels?
    unnested_array = []
    for arrays in array:
        for array_ in arrays:
            unnested_array.append(array_)

    return unnested_array
