'''
Created on 2020-03-24 14:33:48
Last modified on 2020-12-01 13:18:11

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
from abaqusConstants import (TWO_D_PLANAR, DEFORMABLE_BODY, ON,
                             THREE_D, DELETE, GEOMETRY, SUPPRESS)

# standard
from abc import abstractmethod
from abc import ABCMeta

# local library
from .base import Geometry
from .rve_utils_periodic import PeriodicRVEObjInit
from ...utils.utils import unnest


# TODO: handle warnings and errors using particular class
# TODO: base default mesh size in characteristic length
# TODO: use more polymorphic names in methods

# object definition


class RVE(Geometry):
    __metaclass__ = ABCMeta
    bcs_types = {'periodic': PeriodicRVEObjInit}

    def __init__(self, name, dims, center, material, tol, bcs_type, **kwargs):
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
        self.obj_init = self.bcs_types[bcs_type](len(dims), **kwargs)
        self.info = self.obj_init.get_info(name, dims, center, tol)
        self.bcs = self.obj_init.get_bcs(self.info)
        self.obj_creator = self.obj_init.get_obj_creator()
        self.mesh = self.obj_init.get_mesh()

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
                 tol=1e-5, **kwargs):
        dims = (length, width)
        super(RVE2D, self).__init__(name, dims, center, material, tol, bcs_type,
                                    **kwargs)

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
                 bcs_type='periodic', **kwargs):
        super(RVE3D, self).__init__(name, dims, center, material, tol, bcs_type,
                                    **kwargs)

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
