import os
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ensemble_pred_path",
                    required=True,
                    type=str)
parser.add_argument("--ensemble_pred_count",
                    required=True,
                    type=int)
parser.add_argument("--correction_pred_path", 
                    required=True,
                    type=str)
parser.add_argument("--out_dir", 
                    required=True,
                    type=str)

args = parser.parse_args()

ENSEMBLE_PATH = Path(args.ensemble_pred_path)
CORRECTION_PATH = Path(args.correction_pred_path)

df_ens = pd.read_parquet(ENSEMBLE_PATH)
df_cor = pd.read_parquet(CORRECTION_PATH)

a = args.ensemble_pred_count/(args.ensemble_pred_count + 1)
b = 1/(args.ensemble_pred_count + 1)

df_ens["reactivity_2A3_MaP"] = a * df_ens["reactivity_2A3_MaP"] + b * df_cor["reactivity_2A3_MaP"]

df_ens.to_parquet(Path(args.out_dir)/ "final_pred.parquet", index=False)