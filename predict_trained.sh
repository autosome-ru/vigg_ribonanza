#!/bin/bash

bpp_path=data/BPP/eternafold/test
test_path=data/test_sequences.csv
model_weights_dir=model_weights
mkdir -p predictions_trained/single
mkdir -p predictions_trained/main
mkdir -p predictions_trained/final
pred_dir=predictions_trained
bracket_path=data/brackets_test

# models trained on kfold 1000 split without se-blocks and brackets
# total number - 10 models
for i in {0..9}
do
   python3 python_scripts/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/weights_thousands/model_$i.pth --out_path $pred_dir/single --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20
done

# models trained on kfold 1000 split with se-blocks and without brackets
# total number - 15 models
for i in {0..14}
do
   python3 python_scripts/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/se_thousands/model_$i.pth --out_path $pred_dir/single --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20 --use_se
done

# model trained on split by length without se-blocks and without brackets
python3 python_scripts/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/lengths/model_0.pth --out_path $pred_dir/single --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20

# model trained on split by length without se-blocks and with brackets
python3 python_scripts/predict.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/brks_lengths/model_0.pth --out_path $pred_dir/single --device 0 --pos_embedding dyn --adj_ks 3 --num_workers 20 --brackets $bracket_path/ipknot.json $bracket_path/eterna.json

# calculates main prediction by averaging individual predictions made by each model
python3 python_scripts/calculate_main_pred.py --pred_dir $pred_dir/single --out_dir $pred_dir/main
#predicts and adds correction predicted by dms-to-2a3 model
python3 python_scripts/predict_dms22a3.py --bpp_path $bpp_path  --test_path $test_path --model_path $model_weights_dir/dms2a3/model_0.pth --out_path $pred_dir/main --react_preds_path $pred_dir/main/ensemble_pred.parquet  --device 0 --pos_embedding dyn --adj_ks 3 --pred_mode dms_2a3

python3 python_scripts/calculate_final_pred.py --ensemble_pred_path $pred_dir/main/ensemble_pred.parquet --ensemble_pred_count 27 --correction_pred_path $pred_dir/main/submit_dms2a3_model_0.parquet --out_dir $pred_dir/final








