'''
Created on 2020-04-15 14:50:02
Last modified on 2020-05-07 21:21:10
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Develop functions to get data from odb files.
'''


#%% history outputs

def get_xydata_from_nodes_history_output(odb, nodes, variable,
                                         directions=(1, 2, 3),
                                         step_name=None):
    '''
    Given a node array, returns the values for time and the given variable.

    Assumes the variable was requested in history outputs.

    Parameters
    ----------
    odb : Abaqus odb object
    nodes : array-like of Abaqus OdbMeshNode objects
    variable : str
        It only works for vector-like variables. e.g. 'U' and 'RF'
    directions : array
        Directions for which to extract values.
    step_name : str
        If None, it uses the last step.

    Returns
    -------
    y : array-like, shape = [n_nodes x [n_increments x  n_directions]]

    Notes
    -----
    -Assumes all the nodes have history output and the array contains each node
    only once.
    '''

    # initialization
    if not step_name:
        step_name = odb.steps.keys()[-1]
    step = odb.steps[step_name]

    # collect data
    y = []
    for node in nodes:
        instance_name = node.instanceName if node.instanceName else 'ASSEMBLY'
        name = 'Node ' + instance_name + '.' + str(node.label)
        historyOutputs = step.historyRegions[name].historyOutputs
        node_data = []
        for direction in directions:
            node_data.append([data[1] for data in historyOutputs['%s%i' % (variable, direction)].data])
        y.append(node_data)

    return y


#%% field outputs

def get_ydata_from_nodeSets_field_output(odb, nodeSet, variable,
                                         directions=(1, 2, 3), step_name=None,
                                         frames=None):
    '''
    Given a node set, returns the values for the given variable.

    It may take a while to run.

    Parameters
    ----------
    odb : Abaqus odb object
    nodeSet : Abaqus nodeSet object
    variable : str
        It only works for vector-like variables. e.g. 'U' and 'RF'
    directions : array
        Directions for which to extract values. The value will be subtracted
        by 1 when accessing abaqus data.
    frames : array-like of Abaqus OdbFrame objects
        If frames are available from outside, use it because it may significantly
        decreases the computational time.
    step_name : str
        If None, it uses the last step. Only required if frames=None.

    Returns
    -------
    values : array-like, shape = [n_increments x n_directions x n_nodes]]
    '''

    # TODO: extend to scalar and tensor like variables
    # TODO: change name to accept also elements (position should be an input)

    # access frames
    if not frames:
        if not step_name:
            step_name = odb.steps.keys()[-1]
        frames = [frame for frame in odb.steps[step_name].frames]

    # collect data
    values = []
    for frame in frames:
        varFieldOutputs = frame.fieldOutputs[variable]
        outputs = varFieldOutputs.getSubset(region=nodeSet).values
        output_frame = []
        for direction in directions:
            output_frame.append([output.data[direction - 1] for output in outputs])

        values.append(output_frame)

    return values


#%% other

def get_eigenvalues(odb, frames=None, step_name=None):
    '''
    Parameters
    ----------
    odb : Abaqus odb object
    frames : array-like of Abaqus OdbFrame objects
    step_name : str
        If None, it uses the last step. Only required if frames=None.
    '''

    # access frames
    if not frames:
        if not step_name:
            step_name = odb.steps.keys()[-1]
        frames = [frame for frame in odb.steps[step_name].frames]

    # get eigenvalues
    eigenvalues = [float(frame.description.split('EigenValue =')[1]) for frame in list(frames)[1:]]

    return eigenvalues
