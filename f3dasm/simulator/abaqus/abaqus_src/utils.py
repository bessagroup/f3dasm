from collections import OrderedDict

def convert_dict_unicode_str(pickled_dict):
    new_dict = OrderedDict() if type(pickled_dict) is OrderedDict else {}
    for key, value in pickled_dict.items():
        value = _set_converter_flow(value)
        new_dict[str(key)] = value

    return new_dict


def convert_iterable_unicode_str(iterable):
    new_iterable = []
    for value in iterable:
        value = _set_converter_flow(value)
        new_iterable.append(value)

    if type(iterable) is tuple:
        new_iterable = tuple(new_iterable)
    elif type(iterable) is set:
        new_iterable = set(new_iterable)

    return new_iterable


def _set_converter_flow(value):

    if type(value) is unicode:
        value = str(value)
    elif type(value) in [OrderedDict, dict]:
        value = convert_dict_unicode_str(value)
    elif type(value) in [list, tuple, set]:
        value = convert_iterable_unicode_str(value)

    return value

def get_nodes_given_set_names(odb, set_names):
    '''
    Gets array with Abaqus nodes objects given set names.

    Parameters
    ----------
    odb : Abaqus odb object
    set_names : array-like of str

    Returns
    -------
    nodes : array-like of OdbMeshNode objects

    Notes
    -----
    -verifies if nodes are repeated in the different sets.
    '''

    # get all nodes
    nodes_with_reps = []
    for set_name in set_names:
        nodes_with_reps.extend(odb.rootAssembly.nodeSets[set_name].nodes[0])

    # get unique nodes
    if len(set_names) > 1:
        nodes = get_unique_nodes(nodes_with_reps)
    else:
        nodes = nodes_with_reps

    return nodes


def get_unique_nodes(nodes_with_reps):
    '''
    Gets unique nodes in an array of Abaqus OdbMeshNode objects.

    Parameters
    ----------
    nodes_with_reps : array-like of OdbMeshNode objects

    Notes
    -----
    -set data structure does not work with OdbMeshNode objects
    '''

    nodes = []
    for node in nodes_with_reps:
        if node not in nodes:
            nodes.append(node)

    return nodes

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
