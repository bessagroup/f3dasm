'''
Created on 2020-04-25 22:16:33
Last modified on 2020-04-26 01:51:08
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Manipute inputs.
'''


#%% imports

# local library
from ..misc.physics import get_circular_section_props


#%% function definition

def convert_supercompressible(inputs, section=''):
    '''
    Parameters
    ----------
    inputs : dict
        Inputs.
    section : str
        Cross-section geometry. Possible values are 'circular'
    '''

    # initialization
    normalize_by_diameter = ['pitch', 'cross_section_diameter']
    normalize_by_diameter2 = ['area']
    normalize_by_diameter4 = ['Ixx', 'Iyy', 'J']
    normalize_by_diameter_diff = ['top_diameter']
    normalize_by_young_modulus = ['shear_modulus']

    # get new variables
    bottom_diameter = inputs['bottom_diameter']
    young_modulus = inputs['young_modulus']
    new_inputs = {}
    for var_name, variable in inputs.items():

        if var_name[0:5] == 'ratio':

            _, var_name = var_name.split('_', 1)
            if var_name in normalize_by_diameter:
                variable *= bottom_diameter
            elif var_name in normalize_by_diameter2:
                variable *= bottom_diameter**2
            elif var_name in normalize_by_diameter4:
                variable *= bottom_diameter**4
            elif var_name in normalize_by_diameter_diff:
                variable = bottom_diameter * (1 - variable)
            elif var_name in normalize_by_young_modulus:
                variable *= young_modulus

        new_inputs[var_name] = variable

    # add section variables
    if section == 'circular':
        # get diam
        d = new_inputs['cross_section_diameter']
        del new_inputs['cross_section_diameter']

        # get variables
        Ixx, Iyy, J, area = get_circular_section_props(d)

        # save variables
        new_inputs['Ixx'] = Ixx
        new_inputs['Iyy'] = Iyy
        new_inputs['J'] = J
        new_inputs['area'] = area

    return new_inputs
