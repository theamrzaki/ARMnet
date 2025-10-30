#!/bin/bash
# run_ablation.sh
# Script to train full and leave-one-out reward models

# Set Python environment (optional)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

SCRIPT="Mitigation/train2.py"

echo "============================="
echo "Training FULL reward model..."
echo "============================="
python $SCRIPT --experiment full

echo "All experiments finished."


echo "============================="
echo "Creating Plans..."
echo "============================="
python Mitigation/create_plans.py


echo "============================="
echo "Evaluating Plans..."
echo "============================="
python Mitigation/evaluation.py