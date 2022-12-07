#!/bin/bash
#SBATCH --time=0:50:00
#SBATCH --account=def-mpederso
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16G               # memory (per node)
# set name of job
#SBATCH --cpus-per-task=2
#SBATCH --job-name=teldrone_test
#SBATCH --output=tel_test-%J.out
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

python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0001999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0003999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0005999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0007999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0009999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0011999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0013999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0015999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0017999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0019999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0021999.pth
python train_net.py --eval-only --config-file configs/TelDrone-RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/TELDRONE_FPN_CROP/model_0023999.pth