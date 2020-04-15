'''
Created on 2020-04-15 14:50:02
Last modified on 2020-04-15 16:19:06
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Develop functions to get data from odb files.
'''


#%% field outputs

def get_ydata_from_nodeSets_field_output(odb, nodeSets, variable,
                                         directions=(1, 2, 3), step_name=None,
                                         frames=None):
    '''
    Given node sets, returns the values for the given variable.

    It may take a while to run.

    Parameters
    ----------
    odb : Abaqus odb object
    nodeSets : array-like of nodeSets
    variable : str
        It only works for vector-like variables. e.g. 'U' and 'RF'
    directions : array
        Directions for which to extract values.
    frames : array-like of Abaqus OdbFrame objects
        If frames are available from outside, use it because it may significantly
        decreases the computational time.
    step_name : str
        If None, it uses the last step. Only required if frames=None,

    Returns
    -------
    values : array-like, shape = [n_increments]
    '''

    # TODO: extend to scalar and tensor like variables

    # access frames
    if not frames:
        if not step_name:
            step_name = odb.steps.keys()[-1]
        frames = [frame for frame in odb.steps[step_name].frames]

    # collect data
    values = []
    for frame in frames:
        varFieldOutputs = frame.fieldOutputs[variable]
        for nodeSet in nodeSets:
            outputs = varFieldOutputs.getSubset(region=nodeSet).values
            output_frame = []
            for direction in directions:
                output_frame.append([output.data[direction - 1] for output in outputs])

        values.append(output_frame)

    return values
