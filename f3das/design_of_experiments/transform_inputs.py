'''
Created on 2020-04-25 22:16:33
Last modified on 2020-05-07 21:44:29
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Manipute inputs.
'''


#%% function definition

def transform_inputs_supercompressible(inputs):
    '''
    Parameters
    ----------
    inputs : dict
        Inputs.
    section : str
        Cross-section geometry. Possible values are 'circular'
    '''
    # TODO: generalize?

    # initialization
    normalize_by_diameter = ['pitch', 'd']
    normalize_by_diameter2 = ['area']
    normalize_by_diameter4 = ['Ixx', 'Iyy', 'J']
    normalize_by_diameter_diff = ['top_diameter']
    normalize_by_young_modulus = ['shear_modulus']

    # get new variables
    bottom_diameter = inputs['bottom_diameter']
    young_modulus = inputs['young_modulus']
    new_inputs = {}
    for var_name, variable in inputs.items():

        if var_name == 'section':
            continue

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
    if inputs['section'] == 'circular':
        new_inputs['cross_section_props'] = {'type': inputs['section'],
                                             'd': new_inputs['d']}
        del new_inputs['d']

    else:
        new_inputs['cross_section_props'] = {'type': inputs['section'],
                                             'Ixx': new_inputs['Ixx'],
                                             'Iyy': new_inputs['Iyy'],
                                             'J': new_inputs['J'],
                                             'area': new_inputs['area']}
        del new_inputs['Ixx']
        del new_inputs['Iyy']
        del new_inputs['J']
        del new_inputs['area']

    return new_inputs
