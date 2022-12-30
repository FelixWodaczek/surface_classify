#!/bin/bash
#
#----------------------------------------------------------------
# running multiple independent jobs
#----------------------------------------------------------------
#


#  Defining options for slurm how to run
#----------------------------------------------------------------
#
#SBATCH --job-name=arraySurfaceClass
#SBATCH --output=SurfaceClass.log
#
#Number of CPU cores to use within one node
#SBATCH -c 1
#
#Define the number of hours the job should run. 
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=2:00:00
#
#Define the amount of RAM used by your job in GigaBytes
#In shared memory applications this is shared among multiple CPUs
#SBATCH --mem=2G
#
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# load the respective software module(s) you intend to use
#----------------------------------------------------------------
module load python/3.9.7
source SURFCLASS_VENV01/bin/activate
# define sequence of jobs to run as you would do in a BASH script
# use variable $SLURM_ARRAY_TASK_ID to address individual behaviour
# in different iteration of the script execution
#----------------------------------------------------------------

python3 slurm_classify.py -m 'soap_sort' # soap_sort, lmbtr_sort, soap_gendescr, lmbtr_gendescr
deactivate
