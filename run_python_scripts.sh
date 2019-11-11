#!/bin/bash -l
#SBATCH -A dp004
#SBATCH -p cosma7

# Tell SGE that we are using the bash shell
#$ -S /bin/bash

# Give the job a name
#SBATCH --job-name=Python_DI
#SBATCH -t 0-10:00

#SBATCH -o out/slurm.%N.%j.out
#SBATCH -e err/slurm.%N.%j.err

source /etc/profile
shopt -s expand_aliases

module purge

module load utils
module load intel_comp/2018
module load intel_mpi/2018
module load fftw/2.1.5
module load hdf5/1.8.20
module load gsl/2.4
module load hwloc/1.11.11

module unload fftw/2.1.5
module load fftw/3.3.7

module unload python
module load anaconda3/5.2.0

source activate my_python

cd /cosma7/data/dp004/dc-irod1/G-EAGLE/python/ || exit

python Ra_Dec.py
#wait
