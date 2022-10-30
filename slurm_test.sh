#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --account=def-mpederso
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=10G               # memory (per node)
# set name of job
#SBATCH --cpus-per-task=1
#SBATCH --job-name=visdrone_test
#SBATCH --output=vis_test-%J.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=akhilpm135@gmail.com

module load gcc python cuda/11.4 opencv/4.5.5
source ~/envs/detectron2/bin/activate

mkdir  $SLURM_TMPDIR/VisDrone
mkdir  $SLURM_TMPDIR/VisDrone/val
mkdir $SLURM_TMPDIR/VisDrone/annotations

unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/VisDrone2019-DET-val.zip -d $SLURM_TMPDIR
#unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/VisDrone2019-DET-test-dev.zip -d $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/VisDrone2019-DET-val/images/ $SLURM_TMPDIR/VisDrone/val
#cp -r $SLURM_TMPDIR/VisDrone2019-DET-test/images/ $SLURM_TMPDIR/VisDrone/test
cp ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/annotations_VisDrone_val.json $SLURM_TMPDIR/VisDrone/
#cp ~/projects/def-mpederso/akhil135/data_Aerial/VisDrone/annotations_VisDrone_test.json $SLURM_TMPDIR/VisDrone/

#python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN/final.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0004999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0009999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0014999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0019999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0024999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0029999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0034999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0039999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0044999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0049999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0054999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0059999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0064999.pth
python train_net.py --eval-only --config-file configs/RCNN-FPN-CROP.yaml MODEL.WEIGHTS ~/scratch/detectron2/FPN_CROP_LR01_NOP2/model_0069999.pth
