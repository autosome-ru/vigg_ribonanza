import os
import sys
from pathlib import Path
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--train_path",
                    required=True,
                    type=str)
parser.add_argument("--test_path", 
                    required=True,
                    type=str)
parser.add_argument("--arnie_config_path", 
                    required=True,
                    type=str)
parser.add_argument("--arnie_path", 
                    required=True,
                    type=str)
parser.add_argument("--data_dir", 
                    required=True,
                    type=str)
parser.add_argument("--nprocs",
                    default=50,
                    type=int)

args = parser.parse_args()

arnie_config_path = str(Path(args.arnie_config_path).resolve())
os.environ['ARNIEFILE'] = arnie_config_path

arnie_path = str(Path(args.arnie_path).resolve())
sys.path.append(arnie_path)

import arnie
from arnie.bpps import bpps
from arnie.pk_predictors import pk_predict
from arnie.mfe import mfe

import numpy as np 
import pandas as pd
import concurrent.futures
import tqdm

# open train and test datasets
train_data = pd.read_parquet(args.train_path)
test_data = pd.read_csv(args.test_path)



# TRAIN ETERNAFOLD BPPM
print("Calculating: TRAIN ETERNAFOLD BPPM")
AIM_DIR =  Path(args.data_dir) / "BPP" / "eternafold" / "train"
AIM_DIR.mkdir(parents=True, exist_ok=True)

def calc_arnie(seq, seqid):
    res = bpps(seq, package="eternafold")
    outpath = AIM_DIR / f"{seqid}.npy"
    np.save(outpath, res)
    
with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:

    futures = {}
    for ind, seqid, seq in train_data[['sequence_id', 'sequence']].itertuples():
        ft = executor.submit(calc_arnie, seqid=seqid, seq=seq)
        futures[ft] = seqid

    for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            ft.result()
        except Exception as exc:
            print(exc)
            seqid = futures[ft]
            print(f"Error occured while processing {seqid}: {exc}")


        
        
# TEST ETERNAFOLD BPPM
print("Calculating: TEST ETERNAFOLD BPPM")
AIM_DIR =  Path(args.data_dir) / "BPP" / "eternafold" / "test"
AIM_DIR.mkdir(parents=True, exist_ok=True)

with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:

    futures = {}
    for ind, seqid, seq in test_data[['sequence_id', 'sequence']].itertuples():
        ft = executor.submit(calc_arnie, seqid=seqid, seq=seq)
        futures[ft] = seqid

    for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            ft.result()
        except Exception as exc:
            print(exc)
            seqid = futures[ft]
            print(f"Error occured while processing {seqid}: {exc}")
        
# TRAIN ETERNA MFE BRACKETS
print("Calculating: TRAIN ETERNA MFE BRACKETS")
def calc_mfe(seq):
    return mfe(seq, package="eternafold")

seqid2seq = {seqid: seq for ind, seqid, seq in train_data[['sequence_id', 'sequence']].itertuples()}
seq2mfe = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:

    futures = {}
    for seqid, seq in tqdm.tqdm(seqid2seq.items(), total=len(seqid2seq)):
        ft = executor.submit(calc_mfe, seq=seq)
        futures[ft] = seqid


    for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        seqid = futures[ft]
        try:
            r = ft.result()
        except Exception as exc:
            print(exc)

            print(f"Error occured while processing {seqid}: {exc}")
        else:
            seq2mfe[seqid] = r
        

AIM_DIR =  Path(args.data_dir) / "brackets_train" 
AIM_DIR.mkdir(parents=True, exist_ok=True)
with open(AIM_DIR / "eterna.json", "w") as out:
    json.dump(seq2mfe, out)
    

# TEST ETERNA MFE BRACKETS
print("Calculating: TEST ETERNA MFE BRACKETS")
seqid2seq = {seqid: seq for ind, seqid, seq in test_data[['sequence_id', 'sequence']].itertuples()}
seq2mfe = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:

    futures = {}
    for seqid, seq in tqdm.tqdm(seqid2seq.items(), total=len(seqid2seq)):
        ft = executor.submit(calc_mfe, seq=seq)
        futures[ft] = seqid


    for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        seqid = futures[ft]
        try:
            r = ft.result()
        except Exception as exc:
            print(exc)

            print(f"Error occured while processing {seqid}: {exc}")
        else:
            seq2mfe[seqid] = r
        
AIM_DIR =  Path(args.data_dir) / "brackets_test" 
AIM_DIR.mkdir(parents=True, exist_ok=True)
with open(AIM_DIR / "eterna.json", "w") as out:
    json.dump(seq2mfe, out)
    
    
    
# TRAIN IPKNOT BRACKETS
print("Calculating: TRAIN IPKNOT MFE BRACKETS")
def calc_ipknot(seq):
    return pk_predict(seq,'ipknot', refinement=1, cpu=1)

seqid2seq = {seqid: seq for ind, seqid, seq in train_data[['sequence_id', 'sequence']].itertuples()}
seq2mfe = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:

    futures = {}
    for seqid, seq in tqdm.tqdm(seqid2seq.items(), total=len(seqid2seq)):
        ft = executor.submit(calc_ipknot, seq=seq)
        futures[ft] = seqid


    for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        seqid = futures[ft]
        try:
            r = ft.result()
        except Exception as exc:
            print(exc)

            print(f"Error occured while processing {seqid}: {exc}")
        else:
            seq2mfe[seqid] = r
        
AIM_DIR =  Path(args.data_dir) / "brackets_train" 
AIM_DIR.mkdir(parents=True, exist_ok=True)
with open(AIM_DIR / "ipknot.json", "w") as out:
    json.dump(seq2mfe, out)
    
    
# TEST IPKNOT BRACKETS
print("Calculating: TEST IPKNOT MFE BRACKETS")
def calc_ipknot(seq):
    return pk_predict(seq,'ipknot', refinement=1, cpu=1)

# ipknot stalls on some last sequences in test dataset, so we don't include them in calculations
seqid2seq = {seqid: seq for ind, seqid, seq in test_data.iloc[:-20][['sequence_id', 'sequence']].itertuples()}
seq2mfe = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=args.nprocs) as executor:

    futures = {}
    for seqid, seq in tqdm.tqdm(seqid2seq.items(), total=len(seqid2seq)):
        ft = executor.submit(calc_ipknot, seq=seq)
        futures[ft] = seqid


    for ft in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        seqid = futures[ft]
        try:
            r = ft.result()
        except Exception as exc:
            print(exc)

            print(f"Error occured while processing {seqid}: {exc}")
        else:
            seq2mfe[seqid] = r
        
AIM_DIR =  Path(args.data_dir) / "brackets_test" 
AIM_DIR.mkdir(parents=True, exist_ok=True)
with open(AIM_DIR / "ipknot.json", "w") as out:
    json.dump(seq2mfe, out)
    
    

    
