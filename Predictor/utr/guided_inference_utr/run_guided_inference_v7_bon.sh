#!/bin/bash
#SBATCH --job-name=utr_v7_bon
#SBATCH --output=logs/v7_bon_out_%j.log
#SBATCH --error=logs/v7_bon_err_%j.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gvqc0001

mkdir -p logs

module purge
module load python/3.11.9-2cji
module load cuda/12.6.2-yl5o

source ~/evodiff_env/bin/activate

echo "Running on node: $SLURMD_NODENAME"
echo "Launching UTR Guided Inference v7 — Best-of-N"

cd ~/evodiff/Predictor/utr/guided_inference_utr
python -u generate_optimized_utrs.py --config guided_config_v7_best_of_n.yaml

echo "Inference Complete"
