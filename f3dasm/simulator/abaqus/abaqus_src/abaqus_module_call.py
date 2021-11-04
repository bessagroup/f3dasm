import sys
import runpy
import os


def execute(run_module_name, temp_dir, sim_dir ):
    initial_wd = os.getcwd()
    sys.path.append(initial_wd)
    if temp_dir is not None:
        sys.path.append(os.path.join(initial_wd, temp_dir))

    #sim_dir = os.path.join(example_name, sims_dir_name)
    #sim_dir =  os.path.join(sim_dir, 'DoE_point' + str(int(point)))

    os.chdir(sim_dir)
    runpy.run_module(run_module_name, run_name='__main__')
    os.chdir(initial_wd)


# 'import runpy',
# 'import os',
# 'import sys',
# 'initial_wd = os.getcwd()',
# 'sys.path.append(initial_wd)',
# "sys.path.append(os.path.join(initial_wd, '%s'))" % temp_dir_name,
# 'points = %s' % points,
# "sim_dir = r'%s'" % os.path.join(example_name, sims_dir_name),
# 'for point in points:',
# "\tos.chdir('%s' % os.path.join(sim_dir, 'DoE_point%i' % point))",
# "\trunpy.run_module('%s', run_name='__main__')" % run_module_name,
# '\tos.chdir(initial_wd)']


if __name__ == '__main__':
    func =  None
    sim_dir = None
    temp_dir = None

    arglist = sys.argv
    argc  = len(arglist)
    i = 0
    while (i<argc):
        if (arglist[i][:5] == '-func'):
            i +=1
            func = arglist[i]

        elif (arglist[i][:5] == '-tdir'):
            i +=1
            temp_dir = arglist[i]

        elif (arglist[i][:5] == '-sdir'):
            i +=1
            sim_dir = arglist[i]

        i +=1
    execute(func, temp_dir, sim_dir )