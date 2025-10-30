#!/bin/bash

# =========================
# Configurable lists
# =========================
datasets=(
    "re1-ob" "re1-ss" 
    "re2-ob" "re2-ss" 
)

methods=(baro RMDnet)  # include baro
model_class=(Dlinear Fits iTransformer TimeMixerpp) #"iTransformer" "TimeMixerpp" 
scalar_type=("Standard")
combine_baro_options=(true false)
seeds=(1)

# =========================
# Run all valid combinations
# =========================
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for method in "${methods[@]}"; do
            for model in "${model_class[@]}"; do
                for combine_baro in "${combine_baro_options[@]}"; do
                    #print current combination
                    echo "Dataset: $dataset, Method: $method, Model: $model, Combine"
                    # skip baro + combine_baro=true
                    if [ "$method" = "baro" ] && [ "$combine_baro" = true ]; then
                        continue
                    fi

                    #activate conda environment called RCAEval
                    source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate RCAEval
                    cmd="python main.py --dataset $dataset --method $method --seed $seed --model_class $model --scaler_type $scalar_type"

                    if [ "$combine_baro" = true ]; then
                        cmd="$cmd --combine_baro_post"
                    fi

                    echo "Running: $cmd"
                    eval $cmd

                done
            done
        done
    done
done