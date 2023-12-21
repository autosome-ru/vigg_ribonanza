#!/bin/bash

mkdir -p data
cd data
#kaggle competitions download -c stanford-ribonanza-rna-folding -f train_data.csv
#kaggle competitions download -c stanford-ribonanza-rna-folding -f test_sequences.csv

unzip train_data.csv.zip
unzip test_sequences.csv.zip

cd ..

python3 python_scripts/csv_to_parquet.py --file data/train_data.csv --output_file data/train_data.parquet