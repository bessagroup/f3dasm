High-performance computing cluster
----------------------------------

Your `f3dasm`` workflow can be seemlessly translated to a High-performance computing cluster.

You can use the global variable `f3dasm.HPC_JOBID`` to get the job id for an arrayjob
and use it to create a unique output directory for each job.

Example of entire workflow on the HPC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example has been tested with the hpc06 cluster of Delft University of Technology:

Directory Structure:
====================

The directory structure for the project is as follows:

- `my_project/` is the root directory.
- `pbsjob.sh` is the bash script that will be submitted to the HPC.
- `my_script.py` contains the implementation of the `my_function` function.
- `main.py` is the main entry point of the project, governed by `f3dasm`.

.. code-block:: none
   :caption: Directory Structure

   my_project/
   ├── my_script.py
   ├── pbsjob.sh   
   └── main.py

my_script.py
=============

The `my_script.py` file contains your own `my_function` function. You have to modify the function so that it takes a `f3dasm.design` object as input.
retrieves the values of `parameter1` and `parameter2` from the design, performs some calculations or operations, and sets the value of `output1` in the design. 
The function has to return the (modified) design object.


.. code-block:: python
   :caption: my_script.py

    def my_function(design):
        parameter1 = design.get('parameter1')
        parameter2 = design.get('parameter2')
        ...

        design.set('output1', output)
        return design


pbsjob.sh
=========

.. code-block:: bash
   :caption: TORQUE Bash Script

    #!/bin/bash
    # Torque directives (#PBS) must always be at the start of a job script!
    #PBS -N ExampleScript
    #PBS -q mse
    #PBS -l nodes=1:ppn=12,walltime=12:00:00

    # Make sure I'm the only one that can read my output
    umask 0077

    JOB_ID=$(echo "${PBS_JOBID}" | sed 's/\[[^][]*\]//g')

    module load use.own
    module load miniconda3
    cd $PBS_O_WORKDIR

    # Here is where the application is started on the node
    # activating my conda environment:

    source activate f3dasm_env

    # limiting number of threads
    OMP_NUM_THREADS=12
    export OMP_NUM_THREADS=12


    # Check if PBS_ARRAYID exists, else set to 1
    if ! [ -n "${PBS_ARRAYID+1}" ]; then
      PBS_ARRAYID=None
    fi

    #Executing my python program

    python main.py --jobid=${PBS_ARRAYID}



main.py
========

The `main.py` file is the main entry point of the project. 
It imports `f3dasm` and the `my_function` from `my_script.py`. 
In the main function, it creates a design space, fills the design space using a sampler, and executes the data generation function (`my_function`) using the `data.run` method with the specified execution mode.

.. code-block:: python
   :caption: main.py

    import f3dasm
    from my_script import my_function

    # If it is the first job in the array, 
    # first create the designspace, then execute my_function on the designs.
    if f3dasm.HPC_JOBID == 0:
        """Block 1: Design of Experiment"""

        # Create a design space
        design = f3dasm.Domain()

        design.add_input_space(name="parameter1", space=f3dasm.ContinuousParameter(
            lower_bound=0.0, upper_bound=1.0))
        design.add_input_space(name="parameter2", space=f3dasm.ContinuousParameter(
            lower_bound=0.0, upper_bound=1.0))

        design.add_output_space(name="output1", space=f3dasm.ContinuousParameter())

        # Filling the design space
        sampler = f3dasm.sampling.RandomUniform(design)
        data = sampler.get_samples(numsamples=3)

        """Block 2: Data Generation"""

        # Execute the data generation function
        data.run(my_function, mode='cluster')

    # In any other case, the design has already been made
    # Therefore, load it from disk and run my_function on it.
    elif f3dasm.HPC_JOBID > 0:
        # Retrieve the file from disk
        data.from_file()
        data.run(my_function, mode='cluster')


    # Store the data generation function
    data.store()


Run the program
===============

You can run the workflow by submitting the TORQUE file to the HPC queue:

.. code-block:: bash

    qsub pbsjob.sh -t 0-3

The `-t 0-3` option submits an array job with 4 jobs with f3dasm.HPC_JOBID ranging from 0 to 3.
