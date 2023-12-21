import argparse
import pandas as pd

# converts csv to parquet for fatster data loading in next steps 

parser = argparse.ArgumentParser()
parser.add_argument("--file",
                    required=True,
                    type=str)
parser.add_argument("--output_file", 
                    required=True,
                    type=str)

args = parser.parse_args()

df = pd.read_csv(args.file)
df.to_parquet(args.output_file)