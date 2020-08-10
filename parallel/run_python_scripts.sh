#!/bin/bash -l
#SBATCH -p cosma7 # Specify the system.
#SBATCH --ntasks=1 # Set the number of nodes to 1.
#SBATCH --cpus-per-task=8 # Set the number of cores per node to 8 (max is 28 for COSMA7).
#SBATCH -A dp004 # Set the name of the project.
#SBATCH --job-name=Python_DI # Set the name of the job.
#SBATCH -t 0-24:00 # Set a time limit (max is 3 days for COSMA7).
#SBATCH --array=1-50 # Run 50 copies of the code.
# Folders to save the log files #
#SBATCH -o out/slurm.%N.%j.out
#SBATCH -e err/slurm.%N.%j.err

source /etc/profile
shopt -s expand_aliases

module purge

module load gnu_comp/7.3.0
module load openmpi/3.0.1
module load utils
module load gsl/2.4
module load hdf5/1.10.3
module load hwloc/1.11.11

module unload fftw/2.1.5
module load fftw/3.3.7

module unload python
module load anaconda3/5.2.0
source activate my_python

# Run the script 50 times on 1 node with 8 cores.
mpirun -np $SLURM_NTASKS python /cosma7/data/dp004/dc-irod1/EAGLE/python/read_add_attributes.py -rd $SLURM_ARRAY_TASK_ID