# this script searches for model weights in child directories of given source directory and copies them in specified directory

import os
import shutil
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source",
                    required=True,
                    type=str)
parser.add_argument("--out_dir", 
                    required=True,
                    type=str)

args = parser.parse_args()

SOURCE = Path(args.source)
OUT_PATH = Path(args.out_dir)

dirs = os.listdir(SOURCE)
dirpaths = [SOURCE / i for i in dirs if (SOURCE / i).is_dir()]

for path in dirpaths:
    dirs = os.listdir(path)
    modeldirs = [path / i for i in dirs if (path / i).is_dir()]
    for num, model_path in enumerate(modeldirs):
        os.makedirs(OUT_PATH / path.name, exist_ok=True)
        shutil.copy(model_path / 'fst_model' / 'pos_aug_sgd' / 'model.pth', OUT_PATH / path.name / f'model_{num}.pth')