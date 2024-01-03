# Enhancing Contrastive Learning with Noise-Guided Attack: Towards Continual Relation Extraction in the Wild

This repository contains the official PyTorch implementation for the following paper:

> [Enhancing Contrastive Learning with Noise-Guided Attack: Towards Continual Relation Extraction in the Wild](https://arxiv.org/pdf/2305.07085.pdf)

## Environment

Create an environment with the following commands:
```
conda create --name cre python=3.8
conda activate cre
pip install -r requirements.txt
```

## Datasets
All the training and test datasets can be found in the folder `data/`

- **FewRel** relevant data file:
    - data_with_marker.json
    - data_with_marker_train.json
    - data_with_marker_val.json
    - data_with_marker_test.json
    - data_with_marker_train_noise_0.1.json
    - data_with_marker_train_noise_0.3.json
    - data_with_marker_train_noise_0.5.json
    - id2rel.json


- **TACRED** relevant data file:
    - data_with_marker_tacred.json
    - data_with_marker_tacred_train.json
    - data_with_marker_tacred_test.json
    - data_with_marker_tacred_train_noise_0.1.json
    - data_with_marker_tacred_train_noise_0.3.json
    - data_with_marker_tacred_train_noise_0.5.json
    - id2rel_tacred.json


## Sample Commands for Running
```
python -u run_continual.py \
    --gpu 0 \
    --dataname Tacred \
    --lr2 2e-5 \
    --learning_rate 2e-5 \
    --noise_rate 0.1 \ 
    --total_round 1 \
    --hidden \
    --thresh 0.8 \
    --split_steps 3 \
    --margin 0 \
    --amc \
    --temp 0.1
```

