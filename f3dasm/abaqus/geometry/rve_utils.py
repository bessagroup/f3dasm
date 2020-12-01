'''
Created on 2020-12-01 13:09:25
Last modified on 2020-12-01 13:18:40

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''

# imports
from __future__ import division

# abaqus
from caeModules import *  # allow noGui
from part import EdgeArray
from mesh import MeshNodeArray

# standard
from abc import abstractmethod
from abc import ABCMeta
from collections import OrderedDict

# local library
from ...utils.utils import unnest
from ...utils.utils import get_decimal_places


# object definition

class RVEObjInit(object):
    __metaclass__ = ABCMeta

    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def get_info(self, name, dims, center, tol):
        pass

    @abstractmethod
    def get_obj_creator(self):
        pass

    @abstractmethod
    def get_bcs(self, rve_info):
        pass

    @abstractmethod
    def get_mesh(self):
        pass


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
            Possible values are 'vertices', 'edges', 'faces'.
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


class BoundaryConditions(object):
    __metaclass__ = ABCMeta

    @ abstractmethod
    def set_bcs(self, *args, **kwargs):
        pass


class Constraints(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @ abstractmethod
    def create(self):
        pass
