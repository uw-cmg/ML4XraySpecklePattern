#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p sbel_cmg 
#SBATCH --account=skunkworks --qos=skunkworks_owner#priority#skunkworks_owner#courtesy #

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
##SBATCH --gres=gpu:1
#SBATCH -t 0-10:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o matlab-%j.log

## Load CUDA into your environment
## load custimized CUDA and cudaToolkit
module load matlab/r2019b
#module load user/cuda
#module load cuda/10.0
#module load groupmods/cudnn/10.0
#module load gcc/8.2.0
#module load openmpi/4.0.0
#module load intel/compiler
#module load openmpi/intel/2.1.1
#module load openmpi/4.0.0
#module load openmpi/cuda/4.0.0 

#source activate XPCS
#export CUDA_HOME=/usr/local/cuda
#export PATH=$PATH:$CUDA_HOME/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# install chainercv and other libraries for machine learning

# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U scikit-learn
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U tensorflow-gpu
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U keras==2.3.1
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U pandas
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U opencv-python
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U h5py
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U pydot
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U matplotlib
# /srv/home/shenmr/anaconda3/envs/XPCS/bin/pip install -U graphviz

# Running Commands
matlab -nodisplay -r "run('spheres_placing3.m')"
# matlab -nojvm -nodisplay -nosplash < spheres_placing3.m