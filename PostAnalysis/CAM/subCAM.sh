#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p sbel_cmg
#SBATCH --account=cmg --qos=cmg_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
## SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --gres=gpu:1
#SBATCH -t 4-1:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

# activate virtual environment
conda activate keras

# install tensorflow and other libraries for machine learning
# All software installnation should be outside the scripts
module load cuda/10.0


#conda install --name keras tensorflow-gpu=1.13.1
#conda install --name keras keras-gpu=2.2.4

# Should use CMD to install
#pip install git+https://www.github.com/keras-team/keras-contrib.git 
#pip install pydot
#conda install --name keras matplotlib 
#conda install --name keras numpy
#conda install --name keras scipy 
#conda install --name keras pillow
#conda install --name keras scikit-image
#conda install --name keras scikit-learn

# run the training scripts
#python train_resnet50.py --dataset ../../data2 --savedResults saved --trainingLog logs
python genCAM.py
