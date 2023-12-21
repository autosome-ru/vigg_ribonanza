import os
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir",
                    required=True,
                    type=str)
parser.add_argument("--out_dir", 
                    required=True,
                    type=str)

args = parser.parse_args()

PRED_PATH = Path(args.pred_dir)

pred_files = os.listdir(args.pred_dir)

df = pd.read_parquet(PRED_PATH / pred_files[0])
for i in range(1, len(pred_files)):
    temp = pd.read_parquet(PRED_PATH / pred_files[i])
    df["reactivity_DMS_MaP"] = df["reactivity_DMS_MaP"] + temp["reactivity_DMS_MaP"]
    df["reactivity_2A3_MaP"] = df["reactivity_2A3_MaP"] + temp["reactivity_2A3_MaP"]

df["reactivity_DMS_MaP"] = df["reactivity_DMS_MaP"]/len(pred_files)
df["reactivity_2A3_MaP"] = df["reactivity_2A3_MaP"]/len(pred_files)

df.to_parquet(Path(args.out_dir)/ "ensemble_pred.parquet", index=False)
