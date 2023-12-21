# vigg_ribonanza
Source code of the Stanford Ribonanza RNA Folding first place solution

You can read the detailed solution description here -- https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460121

## HARDWARE

Ubuntu 20.04.4 LTS x86_64

CPU: 2x EPYC 7662 (128 CPU each)

6 GPU NVIDIA Tesla V100 32GB CoWoS HBM2

1000GB RAM

2TB SSD

## Installation steps 

Use `setup_data.sh` to download train and test data 

Use `setup_tools.sh` to install all required software

Use `calculate_features.sh` to calculate features required for the model 

## How to train your model

To train model use `python_scripts/train_uni_adjnet.py` script 

We recommend to train model with the following options

```
python3 python_scripts/train_uni_adjnet_se.py --bpp_path /projects_nvme/deepbeer/eterna/ --train_path /projects/deepbeer/ribonanza/train_data/train_data.parquet  --out_path  outmodel_dir --device 0 --num_workers 20 --wd 0.05 --epoch 270 --lr_max 5e-3 --pct_start 0.05 --batch_cnt 1791 --sgd_lr 5e-5 --sgd_epochs 25 --sgd_batch_cnt 500 --sgd_wd 0.05 --fold 0 --nfolds 1000 --pos_embedding dyn --adj_ks 3 --seed 42 --use_se
```

## How to make predictions on a new test set.

Use `python_scripts/train_uni_adjnet.py`

Example:

```
python3 python_scripts/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_path --out_path $out_dir --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20 --use_se
```

## Contacts

In case of any question you write to dmitrypenzar1996@gmail.com