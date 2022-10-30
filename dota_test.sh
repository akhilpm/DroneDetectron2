#!/bin/bash
#SBATCH --time=4:50:00
#SBATCH --account=def-mpederso
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=16G               # memory (per node)
# set name of job
#SBATCH --cpus-per-task=2
#SBATCH --job-name=dota_test
#SBATCH --output=dota_test-%J.out
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=akhilpm135@gmail.com

module load gcc python cuda/11.4 opencv/4.5.5
source ~/envs/detectron2/bin/activate

mkdir  $SLURM_TMPDIR/DOTA
mkdir  $SLURM_TMPDIR/DOTA/val
unzip -q ~/projects/def-mpederso/akhil135/data_Aerial/DOTA/DOTA-val.zip -d $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/DOTA-val/images/ $SLURM_TMPDIR/DOTA/val
cp ~/projects/def-mpederso/akhil135/data_Aerial/DOTA/annotations_DOTA_val.json $SLURM_TMPDIR/DOTA/

python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0004999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0009999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0014999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0019999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0024999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0029999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0034999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0039999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0044999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0049999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0054999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0059999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0064999.pth
python train_net.py --eval-only --config-file configs/Dota-Base-RCNN-FPN.yaml MODEL.WEIGHTS ~/scratch/detectron2/DOTA_FPN/model_0069999.pth