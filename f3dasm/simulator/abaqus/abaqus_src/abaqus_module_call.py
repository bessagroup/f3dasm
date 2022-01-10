import sys
import runpy
import os

def execute(run_module_name, temp_dir, sim_dir ):
    initial_wd = os.getcwd()
    sys.path.append(initial_wd)
    if temp_dir is not None:
        sys.path.append(os.path.join(initial_wd, temp_dir))

    os.chdir(sim_dir)
    runpy.run_module(run_module_name, run_name='__main__')
    os.chdir(initial_wd)


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