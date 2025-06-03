#!/bin/bash
# run_training.sh

# Set the specific GPU to use (e.g., GPU 3)
export CUDA_VISIBLE_DEVICES=0

# prompt1 pvq21 indomain format omitted 1epoch
python3 train.py --config config/generator_prompt1_pvq21_indomain_over7_1epoch.yaml
python3 inference.py --config config/generator_prompt1_pvq21_indomain_over7_1epoch.yaml


# # prompt1 pvq21 indomain format omitted 5epoch
python3 train.py --config config/generator_prompt1_pvq21_indomain_over7_5epoch.yaml
python3 inference.py --config config/generator_prompt1_pvq21_indomain_over7_5epoch.yaml
