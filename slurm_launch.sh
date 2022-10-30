#!/bin/bash
#SBATCH --time=18:50:00
#SBATCH --account=def-mpederso
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16G               # memory (per node)
# set name of job
#SBATCH --cpus-per-task=2
#SBATCH --job-name=visdrone_train
#SBATCH --output=vis_train-%J.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=akhilpm135@gmail.com

module load gcc python cuda/11.4 opencv/4.5.5
source ~/envs/detectron2/bin/activate

mkdir  $SLURM_TMPDIR/VisDrone
mkdir  $SLURM_TMPDIR/VisDrone/train

unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/VisDrone2019-DET-train.zip -d $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/VisDrone2019-DET-train/images/ $SLURM_TMPDIR/VisDrone/train
cp ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/annotations_VisDrone_train.json $SLURM_TMPDIR/VisDrone/

mkdir  $SLURM_TMPDIR/VisDrone/val
unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/VisDrone2019-DET-val.zip -d $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/VisDrone2019-DET-val/images/ $SLURM_TMPDIR/VisDrone/val
cp ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/annotations_VisDrone_val.json $SLURM_TMPDIR/VisDrone/
#cp ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/train_crops.pkl $SLURM_TMPDIR/VisDrone/

python train_net.py --num-gpus 1 --config-file configs/RCNN-FPN-CROP.yaml OUTPUT_DIR ~/scratch/detectron2/FPN_CROP_LR01_NOP2