"""
Using hydra on the high-performance cluster computer
====================================================

`hydra <https://hydra.cc/>`_ can be seamlessly integrated with the worfklows in :mod:`f3dasm` to manage the configuration settings for the project.
"""
###############################################################################
#
# The following example is the same as in section :ref:`workflow`; we will create a workflow for the following data-driven process:
#
# * Create a 2D continuous :class:`~f3dasm.design.Domain`
# * Sample from the domain using a the Latin-hypercube sampler
# * Use a data generation function, which will be the ``"Ackley"`` function a from the :ref:`benchmark-functions`
#
# .. image:: ../../img/f3dasm-workflow-example-cluster.png
#    :width: 70%
#    :align: center
#    :alt: Workflow

from time import sleep

import hydra

from f3dasm import ExperimentData

###############################################################################
# Directory Structure
# ^^^^^^^^^^^^^^^^^^^
#
# The directory structure for the project is as follows:
#
# - `my_project/` is the current working directory.
# - `config.yaml` is a hydra YAML configuration file.
# - `main.py` is the main entry point of the project, governed by :mod:`f3dasm`.
#
#
# .. code-block:: none
#    :caption: Directory Structure
#
#    my_project/
#    ├── my_script.py
#    └── config.yaml
#    └── main.py
#
# The `config_from_sampling.yaml` file contains the configuration settings for the project:
#
# .. code-block:: yaml
#    :caption: config_from_sampling.yaml
#
#         domain:
#         x0:
#             type: float
#             low: 0.
#             high: 1.
#         x1:
#             type: float
#             low: 0.
#             high: 1.
#
#         experimentdata:
#         from_sampling:
#             domain: ${domain}
#             sampler: random
#             seed: 1
#             n_samples: 10
#
#         mode: sequential
#
#         hpc:
#         jobid: -1
#
# It specifies the search-space domain, sampler settings, and the execution mode (`sequential` in this case).
# The domain is defined with `x0` and `x1` as continuous parameters with their corresponding lower and upper bounds.
#
# We want to make sure that the sampling is done only once, and that the data generation is done in parallel.
# Therefore we can divide the different nodes into two categories:
#
# * The first node (:code:`f3dasm.HPC_JOBID == 0`) will be the **master** node, which will be responsible for creating the design-of-experiments and sampling (the ``create_experimentdata`` function).


def create_experimentdata(config):
    """Design of Experiment"""
    # Create the ExperimentData object
    data = ExperimentData.from_yaml(config.experimentdata)

    # Store the data to disk
    data.store()


def worker_node(config):
    # Extract the experimentdata from disk
    data = ExperimentData.from_file(project_dir='.')

    """Data Generation"""
    # Use the data-generator to evaluate the initial samples
    data.evaluate(data_generator='Ackley', mode=config.mode)


###############################################################################
# The entrypoint of the script can now check the jobid of the current node and decide whether to create the experiment data or to run the data generation function:

@hydra.main(config_path=".", config_name="config_from_sampling")
def main(config):
    # Check the jobid of the current node
    if config.hpc.jobid == 0:
        create_experimentdata()
        worker_node()
    elif config.hpc.jobid == -1:  # Sequential
        create_experimentdata()
        worker_node()
    elif config.hpc.jobid > 0:
        # Asynchronize the jobs in order to omit racing conditions
        sleep(config.hpc.jobid)
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
#             python main.py ++hpc.jobid=${PBS_ARRAYID} hydra.run.dir=outputs/${now:%Y-%m-%d}/${JOB_ID}
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
#             python main.py ++hpc.jobid=${SLURM_ARRAY_TASK_ID} hydra.run.dir=/scratch/${USER}/${projectfolder}/${SLURM_ARRAY_JOB_ID}
#
# .. warning::
#     Make sure you set the ``hydra.run.dir`` argument in the jobscript to the location where you want to store the output of the hydra runs!
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
