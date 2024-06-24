#!/bin/sh
### General options
### -- specify queue --
###BSUB -q hpc
#BSUB -q gpuv100
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### –- specify queue --
### -- set the job Name --
#BSUB -J training_ta_100
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need xGB of memory per core/slot --
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
###BSUB -M 128GB
### -- set walltime limit: hh:mm --
#BSUB -W 05:00
### -- set the email address --
#BSUB -u s214205@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- set the job output file --
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o training_ta_100_%J.out
#BSUB -e training_ta_100_%J.err
# all  BSUB option comments should be above this line!

# execute our command
source ~/.bashrc
conda activate spindle
module load cuda/12.2
module load cudnn/v8.9.7.29-prod-cuda-12.X
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/zhome/01/4/167733/anaconda3/lib/
python /zhome/01/4/167733/autoencoder_code/train_ta_100.py