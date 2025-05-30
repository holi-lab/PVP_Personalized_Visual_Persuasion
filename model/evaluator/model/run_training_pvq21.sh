#!/bin/bash
# run_training.sh

# Set the specific GPU to use (e.g., GPU 3)
export CUDA_VISIBLE_DEVICES=3

## prompt1 pvq21 indomain 1epoch
python3 train.py --config config/prompt1_pvq21_indomain_1epoch.yaml
python3 inference.py --config config/prompt1_pvq21_indomain_1epoch.yaml
# python3 metric.py --config config/prompt1_pvq21_indomain_1epoch_category.yaml
# python3 metric_re.py --config config/prompt1_pvq21_indomain_1epoch_category.yaml