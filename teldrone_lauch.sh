#!/bin/bash
#SBATCH --time=4:50:00
#SBATCH --account=def-mpederso
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16G               # memory (per node)
# set name of job
#SBATCH --cpus-per-task=2
#SBATCH --job-name=teldrone_train
#SBATCH --output=tel_train-%J.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=akhilpm135@gmail.com

module load gcc python cuda/11.4 opencv/4.5.5
source ~/envs/detectron2/bin/activate

mkdir $SLURM_TMPDIR/TelDrone
mkdir $SLURM_TMPDIR/TelDrone/train
mkdir $SLURM_TMPDIR/TelDrone/train/images

unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/TelDrone/AngmeringMay2020.zip -d $SLURM_TMPDIR
cp $SLURM_TMPDIR/AngmeringMay2020/*  $SLURM_TMPDIR/TelDrone/train/images
cp ~/projects/def-mpederso/akhil135/data_Aerial/TelDrone/annotations_TelDrone_train.json $SLURM_TMPDIR/TelDrone/

#python train_net.py --num-gpus 1 --config-file configs/TelDrone-RCNN-FPN.yaml OUTPUT_DIR ~/scratch/detectron2/TELDRONE_FPN
python train_net.py --num-gpus 1 --config-file configs/TelDrone-RCNN-FPN-CROP.yaml OUTPUT_DIR ~/scratch/detectron2/TELDRONE_FPN_CROP