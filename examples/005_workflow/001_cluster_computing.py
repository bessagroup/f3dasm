"""
Using f3dasm on a high-performance cluster computer
===================================================

Your :mod:`f3dasm` workflow can be seemlessly translated to a high-performance computing cluster.
The advantage is that you can parallelize the total number of experiments among the nodes of the cluster.
This is especially useful when you have a large number of experiments to run.
"""

###############################################################################
# .. note::
#     This example has been tested on the following high-performance computing cluster systems:
#
#     * The `hpc06 cluster of Delft University of Technology <https://hpcwiki.tudelft.nl/index.php/Main_Page>`_ , using the `TORQUE resource manager <https://en.wikipedia.org/wiki/TORQUE>`_.
#     * The `DelftBlue: TU Delft supercomputer <https://www.tudelft.nl/dhpc/system>`_, using the `SLURM resource manager <https://slurm.schedmd.com/documentation.html>`_.
#     * The `OSCAR compute cluster from Brown University <https://docs.ccv.brown.edu/oscar/getting-started>`_, using the `SLURM resource manager <https://slurm.schedmd.com/documentation.html>`_.

from time import sleep

import numpy as np

from f3dasm import HPC_JOBID, ExperimentData
from f3dasm.design import make_nd_continuous_domain

###############################################################################
# The following example is the same as in section :ref:`workflow`; only now we are omiting the optimization part and only parallelize the data generation:
#
# * Create a 20D continuous :class:`~f3dasm.design.Domain`
# * Sample from the domain using a the Latin-hypercube sampler
# * With multiple nodes; use a data generation function, which will be the ``"Ackley"`` function a from the :ref:`benchmark-functions`
#
#
# .. image:: ../../img/f3dasm-workflow-example-cluster.png
#    :width: 70%
#    :align: center
#    :alt: Workflow

###############################################################################
# We want to make sure that the sampling is done only once, and that the data generation is done in parallel.
# Therefore we can divide the different nodes into two categories:
#
# * The first node (:code:`f3dasm.HPC_JOBID == 0`) will be the **master** node, which will be responsible for creating the design-of-experiments and sampling (the ``create_experimentdata`` function).


def create_experimentdata():
    """Design of Experiment"""
    # Create a domain object
    domain = make_nd_continuous_domain(
        bounds=np.tile([0.0, 1.0], (20, 1)), dimensionality=20)

    # Create the ExperimentData object
    data = ExperimentData(domain=domain)

    # Sampling from the domain
    data.sample(sampler='latin', n_samples=10)

    # Store the data to disk
    data.store()

###############################################################################
# * All the other nodes (:code:`f3dasm.HPC_JOBID > 0`) will be **process** nodes, which will retrieve the :class:`~f3dasm.ExperimentData` from disk and go straight to the data generation function.
#
# .. image:: ../../img/f3dasm-workflow-cluster-roles.png
#    :width: 100%
#    :align: center
#    :alt: Cluster roles


def worker_node():
    # Extract the experimentdata from disk
    data = ExperimentData.from_file(project_dir='.')

    """Data Generation"""
    # Use the data-generator to evaluate the initial samples
    data.evaluate(data_generator='Ackley', mode='cluster')

###############################################################################
# The entrypoint of the script can now check the jobid of the current node and decide whether to create the experiment data or to run the data generation function:


if __name__ == '__main__':
    # Check the jobid of the current node
    if HPC_JOBID is None:
        # If the jobid is none, we are not running anything now
        pass

    elif HPC_JOBID == 0:
        create_experimentdata()
        worker_node()
    elif HPC_JOBID > 0:
        # Asynchronize the jobs in order to omit racing conditions
        sleep(HPC_JOBID)
        worker_node()

###############################################################################
#
# Running the program
# -------------------
#
# You can run the workflow by submitting the bash script to the HPC queue:
# Make sure you have `miniconda3 <https://docs.anaconda.com/free/miniconda/index.html>`_ installed on the cluster, and that you have created a conda environment (in this example named ``f3dasm_env``) with the necessary packages:
#
# .. tabs::
#
#     .. group-tab:: TORQUE
#
#         .. code-block:: bash
#
#             #!/bin/bash
#             # Torque directives (#PBS) must always be at the start of a job script!
#             #PBS -N ExampleScript
#             #PBS -q mse
#             #PBS -l nodes=1:ppn=12,walltime=12:00:00
#
#             # Make sure I'm the only one that can read my output
#             umask 0077
#
#
#             # The PBS_JOBID looks like 1234566[0].
#             # With the following line, we extract the PBS_ARRAYID, the part in the brackets []:
#             PBS_ARRAYID=$(echo "${PBS_JOBID}" | sed 's/\[[^][]*\]//g')
#
#             module load use.own
#             module load miniconda3
#             cd $PBS_O_WORKDIR
#
#             # Here is where the application is started on the node
#             # activating my conda environment:
#
#             source activate f3dasm_env
#
#             # limiting number of threads
#             OMP_NUM_THREADS=12
#             export OMP_NUM_THREADS=12
#
#
#             # If the PBS_ARRAYID is not set, set it to None
#             if ! [ -n "${PBS_ARRAYID+1}" ]; then
#             PBS_ARRAYID=None
#             fi
#
#             # Executing my python program with the jobid flag
#             python main.py --jobid=${PBS_ARRAYID}
#
#     .. group-tab:: SLURM
#
#         .. code-block:: bash
#
#             #!/bin/bash -l
#
#             #SBATCH -J "ExmpleScript"            		# name of the job (can be change to whichever name you like)
#             #SBATCH --get-user-env             			# to set environment variables
#
#             #SBATCH --partition=compute
#             #SBATCH --time=12:00:00
#             #SBATCH --nodes=1
#             #SBATCH --ntasks-per-node=12
#             #SBATCH --cpus-per-task=1
#             #SBATCH --mem=0
#             #SBATCH --account=research-eemcs-me
#             #SBATCH --array=0-2
#
#             source activate f3dasm_env
#
#             # Executing my python program with the jobid flag
#             python3 main.py --jobid=${SLURM_ARRAY_TASK_ID}
#
#
# You can run the workflow by submitting the bash script to the HPC queue.
# the following command submits an array job with 3 jobs with :code:`f3dasm.HPC_JOBID` of 0, 1 and 2.
#
# .. tabs::
#
#     .. group-tab:: TORQUE
#
#         .. code-block:: bash
#
#             qsub pbsjob.sh -t 0-2
#
#     .. group-tab:: SLURM
#
#         .. code-block:: bash
#
#             sbatch --array 0-2 pbsjob.sh
#
