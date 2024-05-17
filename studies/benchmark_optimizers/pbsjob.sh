#!/bin/bash
# Torque directives (#PBS) must always be at the start of a job script!
#PBS -N CreateTrainingData
#PBS -q mse
#PBS -l nodes=1:ppn=12,walltime=12:00:00
# Set the name of the job
#
# Set the rerunable flag, 'n' is not rerunable, default is 'y'

# Make sure I'm the only one that can read my output
umask 0077

# ####################################################
# # OPTIONAL: creating temporary directory on the node
# # for details see:
# # https://hpcwiki.tudelft.nl/index.php/More_about_queues_and_nodes

# # create a temporary directory in /var/tmp
# TMP=/var/tmp/${PBS_JOBID}
# mkdir -p ${TMP}
# echo "Temporary work dir: ${TMP}"
# if [ ! -d "${TMP}" ]; then
#     echo "Cannot create temporary directory. Disk probably full."
#     exit 1
# fi

# # copy the input files to ${TMP}
# echo "Copying from ${PBS_O_WORKDIR}/ to ${TMP}/"
# /usr/bin/rsync -vax "${PBS_O_WORKDIR}/" ${TMP}/
# cd ${TMP}
# #################################################### 

JOB_ID=$(echo "${PBS_JOBID}" | sed 's/\[[^][]*\]//g')

module load use.own
module load miniconda3
cd $PBS_O_WORKDIR

# Here is where the application is started on the node
# activating my conda environment:

source activate f3dasm_env

# limiting number of threads -> see hpcwiki

OMP_NUM_THREADS=12
export OMP_NUM_THREADS=12


# Check if PBS_ARRAYID exists, else set to -1
if ! [ -n "${PBS_ARRAYID+1}" ]; then
  PBS_ARRAYID=-1
fi

#Executing my python program

python main.py ++hpc.jobid=${PBS_ARRAYID} hydra.run.dir=outputs/${now:%Y-%m-%d}/${JOB_ID}

# job done, copy everything back
# echo "Copying from ${TMP}/ to ${PBS_O_WORKDIR}/"
# /usr/bin/rsync -vax ${TMP}/ "${PBS_O_WORKDIR}/"

# # delete my temporary files
# [ $? -eq 0 ] && /bin/rm -rf ${TMP} 