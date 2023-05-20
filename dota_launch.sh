#!/bin/bash
#SBATCH --time=9:50:00
#SBATCH --account=def-mpederso
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=24G               # memory (per node)
# set name of job
#SBATCH --cpus-per-task=4
#SBATCH --job-name=dota_train
#SBATCH --output=dota_train-%J.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=akhilpm135@gmail.com

module load gcc python cuda/11.4 opencv/4.5.5
source ~/envs/detectron2/bin/activate

mkdir  $SLURM_TMPDIR/DOTA
mkdir  $SLURM_TMPDIR/DOTA/train
unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/DOTA/DOTA-train.zip -d $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/DOTA-train/images/ $SLURM_TMPDIR/DOTA/train
cp ~/projects/def-mpederso/akhil135/data_Aerial/DOTA/annotations_DOTA_train.json $SLURM_TMPDIR/DOTA/

mkdir  $SLURM_TMPDIR/DOTA/val
unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/DOTA/DOTA-val.zip -d $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/DOTA-val/images/ $SLURM_TMPDIR/DOTA/val
cp ~/projects/def-mpederso/akhil135/data_Aerial/DOTA/annotations_DOTA_val.json $SLURM_TMPDIR/DOTA/

#python train_net.py --resume --num-gpus 1 --config-file configs/Dota-RCNN-FPN-CROP.yaml OUTPUT_DIR ~/scratch/detectron2/DOTA_CROP_1
python train_net.py --resume --num-gpus 1 --config-file configs/Dota-Base-RCNN-FPN.yaml OUTPUT_DIR ~/scratch/detectron2/DOTA_1
#python train_net.py --eval-only --num-gpus 1 --config-file configs/Dota-RCNN-FPN-CROP.yaml MODEL.WEIGHTS /home/akhil135/scratch/detectron2/DOTA_CROP_10/model_0008999.pth