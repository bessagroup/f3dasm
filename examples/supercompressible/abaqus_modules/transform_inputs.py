'''
Created on 2020-09-29 11:09:15
Last modified on 2020-09-29 11:09:53

@author: L. F. Pereira (lfpereira@fe.up.pt))
'''


def transform_inputs_supercompressible(inputs):
    # initialization
    normalize_by_diameter = ['pitch', 'd']
    normalize_by_diameter_diff = ['top_diameter']

    # get new variables
    bottom_diameter = inputs['bottom_diameter']
    new_inputs = {}
    for var_name, variable in inputs.items():
        if var_name[0:5] == 'ratio':

            _, var_name = var_name.split('_', 1)
            if var_name in normalize_by_diameter:
                variable *= bottom_diameter
            elif var_name in normalize_by_diameter_diff:
                variable = bottom_diameter * (1 - variable)

        new_inputs[var_name] = variable

    return new_inputs
