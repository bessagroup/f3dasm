'''
Created on 2020-04-25 15:56:27
Last modified on 2020-09-29 09:06:04

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Define functions used to manipulate files.
'''


# imports

# standard library
from __future__ import print_function
import os
import glob
from collections import OrderedDict

# third-party
import numpy as np

# local library
from .utils import get_int_number_from_str


# function definition

def collect_folder_names(sim_dir, sim_numbers=None):
    '''
    Collect simulation folders.

    Parameters
    ----------
    sim_dir : str
        Analyses directory.
    sim_numbers : array-like
        Number of the simulations that must be considered. If None, all the
        folders are considered.

    Notes
    -----
    -Directory must contain only folders related with simulations.
    '''
    # TODO: generalize for other kinds of files

    folder_names_temp = [name for name in os.listdir(sim_dir)
                         if os.path.isdir(os.path.join(sim_dir, name))]
    existing_sim_numbers = [get_int_number_from_str(folder_name)
                            for folder_name in folder_names_temp]
    indices = sorted(range(len(existing_sim_numbers)), key=existing_sim_numbers.__getitem__)
    if sim_numbers is None:
        sim_numbers = existing_sim_numbers
        sim_numbers.sort()
    folder_names = [folder_names_temp[index] for index in indices
                    if existing_sim_numbers[index] in sim_numbers]

    return folder_names


def get_unique_file_by_ext(dir_name=None, ext='.pkl'):
    '''
    Get name of a file.

    Notes
    -----
    -Assumes there's only one file with the given extension in the directory.
    '''

    # initialization
    if not dir_name:
        dir_name = os.getcwd()

    # get file name
    for fname in os.listdir(dir_name):
        if fname.endswith(ext):
            break
    else:
        raise Exception('File not found')

    return fname


def verify_existing_name(name_init):
    try:
        name, ext = os.path.splitext(name_init)
    except ValueError:
        name = name_init
        ext = ''
    filename = name_init
    i = 1
    while os.path.exists(filename):
        i += 1
        filename = name + '_%s' % str(i) + ext

    return filename


def get_sorted_by_time(parcial_name, dir_path=None, ext=None):
    '''
    Gets the most recent created folder that contains part of a given name.
    '''

    # initialization
    dir_path = os.getcwd() if dir_path is None else dir_path
    potential_file_names = []
    created_times = []

    # get created times
    if ext is None:
        potential_file_names = [name for name in os.listdir(dir_path)
                                if os.path.isdir(name) and parcial_name in name]
    else:
        potential_file_names = [name for name in os.listdir(dir_path)
                                if name.endswith('.' + ext) and parcial_name in name]
    created_times = [os.path.getctime(os.path.join(dir_path, name))
                     for name in potential_file_names]
    indices = np.argsort(created_times)

    # find most recent folder
    filenames = [potential_file_names[index] for index in indices]

    return filenames


def clean_abaqus_dir(ext2rem=('.abq', '.com', '.log', '.mdl', '.pac', '.rpy',
                              '.sel', '.stt'),
                     dir_path=None):

    # local functions
    def rmfile(ftype):
        file_list = glob.glob(os.path.join(dir_path, '*' + ftype))
        for file in file_list:
            try:
                os.remove(file)
            except:
                pass

    # initialization
    dir_path = os.getcwd() if dir_path is None else dir_path

    # deleting files
    for ftype in ext2rem:
        rmfile(ftype)

    # delete another .rpy
    if '.rpy' in ext2rem:
        ftype = 'rpy.*'
        rmfile(ftype)


class InfoReport(object):

    def __init__(self, sections=None):
        self.sections = OrderedDict()
        if sections is not None:
            for section in sections:
                if type(section) is str:
                    self.add_section(section)
                else:
                    self.add_section(section[0], section[1])

    class Section(object):

        def __init__(self, name, header=''):
            self.name = name
            self.header = header
            self.info = []

        def add_info(self, info):
            self.info.append(info)

    def __getitem__(self, attr):
        return self.sections[attr]

    def append(self, other):
        for section_name in self.sections.keys():
            if section_name in other.sections.keys():
                raise Exception('Sections have the same name')

        self.sections.update(other.sections)

    def add_section(self, name, header=None):
        self.sections[name] = self.Section(name, header)

    def add_info(self, section, info):
        self.sections[section].add_info(info)

    def print_info(self, print_headers=True, sections_split='\n'):
        for i, section in enumerate(self.sections.values()):
            if print_headers and section.header:
                print(section.header)
            for info in section.info:
                print(info)
            if sections_split and i < len(self.sections) - 1:
                print(sections_split)

    def write_report(self, file, print_headers=True, sections_split='\n'):
        for i, section in enumerate(self.sections.values()):
            if print_headers and section.header:
                file.write('{}\n'.format(section.header))
            for info in section.info:
                file.write('{}\n'.format(info))
            if sections_split and i < len(self.sections) - 1:
                file.write(sections_split)
