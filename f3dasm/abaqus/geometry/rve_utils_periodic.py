'''
Created on 2020-12-01 13:09:44
Last modified on 2020-12-01 13:21:57

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports
from __future__ import division

# abaqus
from caeModules import *  # allow noGui
from abaqusConstants import (FIXED, TET, FREE, YZPLANE, XZPLANE, XYPLANE)
from part import FaceArray
from mesh import MeshNodeArray

# standard
from abc import abstractmethod
from abc import ABCMeta

# local library
from .rve_utils import RVEObjInit
from .rve_utils import RVEInfo2D
from .rve_utils import RVEInfo3D
from .rve_utils import RVEObjCreator
from .rve_utils import BoundaryConditions
from .rve_utils import Constraints
from ..modelling.bcs import DisplacementBC
from ..modelling.mesh import MeshGenerator
from ...utils.linalg import symmetricize_vector
from ...utils.solid_mechanics import compute_small_strains_from_green
from ...utils.utils import unnest


# object definition

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

    def __init__(self, mesh_checker='by_closest'):
        '''
        Parameters
        ----------
        mesh_checker : str
            Possible values are 'by_closest', 'by_sorting'.
        '''
        super(PeriodicMeshGenerator3D, self).__init__()
        if mesh_checker == 'by_closest':
            self.mesh_checker = PeriodicMeshChecker3DByClosest()
        else:
            self.mesh_checker = PeriodicMeshChecker3DBySorting()

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
    __metaclass__ = ABCMeta

    def __init__(self):
        super(PeriodicMeshChecker3D, self).__init__()
        self.face_by_closest = False

    @abstractmethod
    def _verify_faces(self, rve_info):
        pass

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
        return self._verify_faces(rve_info)

    def _verify_tol_face_nodes(self, rve_info, node, node_cmp, j, k):
        if abs(node.coordinates[j] - node_cmp.coordinates[j]) > rve_info.tol or abs(node.coordinates[k] - node_cmp.coordinates[k]) > rve_info.tol:
            # create set with error nodes
            set_name = rve_info.verify_set_name('_ERROR_FACE_NODES')
            rve_info.part.Set(set_name, nodes=MeshNodeArray((node, node_cmp)))
            return False

        return True


class PeriodicMeshChecker3DByClosest(PeriodicMeshChecker3D):

    def _verify_faces(self, rve_info):

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


class PeriodicMeshChecker3DBySorting(PeriodicMeshChecker3D):

    def _verify_faces(self, rve_info):
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


class PeriodicRVEObjInit(RVEObjInit):
    mesh_strats = {'simple': PeriodicMeshGenerator3DSimple,
                   'S1': PeriodicMeshGenerator3DS1}

    def __init__(self, dim, mesh_strat='simple', mesh_checker='by_closest'):
        '''
        Parameters
        ----------
        mesh_strat : str
            Possible values are 'simple', 'S1'. Only applicable to `dim == 3`.
        mesh_checker : str
            Possible values are 'by_closest', 'by_sorting'. Only applicable
            to `dim == 3`.
        '''
        super(PeriodicRVEObjInit, self).__init__(dim)
        self.mesh_strat = mesh_strat
        self.mesh_checker = mesh_checker

    def get_info(self, name, dims, center, tol):
        if self.dim == 2:
            return PeriodicRVEInfo2D(name, dims, center, tol)
        else:
            return PeriodicRVEInfo3D(name, dims, center, tol)

    def get_bcs(self, rve_info):
        return PeriodicBoundaryConditions(rve_info)

    def get_obj_creator(self):
        return PeriodicRVEObjCreator()

    def get_mesh(self):
        if self.dim == 2:
            return PeriodicMeshGenerator2D()
        else:
            return self.mesh_strats[self.mesh_strat](mesh_checker=self.mesh_checker)
