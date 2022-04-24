from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import sys

sys.path.append('/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks')

# Flag to append either classification or regression labels
regression = True

def smile_to_fingerprint(smile, radius, bitlength):
    mol = Chem.MolFromSmiles(smile)
    return AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=bitlength).ToList()

def fingerprints_from_smiles_dataset(data, regression=False):
    rows = []
    for index, row in data.iterrows():
        fingerprint = smile_to_fingerprint(row['mol'], 2, 1024)
        if regression:
            new_row = [row["pIC50"]] + fingerprint
        else:
            new_row = [row["Class"]] + fingerprint
        rows.append(new_row)
    data = pd.DataFrame(rows)
    headers = data.columns.values.tolist()
    if regression:
        data = data.rename(columns={headers[0]:"pIC50"})
    else:
        data = data.rename(columns={headers[0]:"Class"})
    return data

def generate_fingerprints(regression=False):
    train_df = pd.read_csv("/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/split_bace/train/bace/raw/bace.csv")
    test_df = pd.read_csv("/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/split_bace/test/bace/raw/bace.csv")
    if regression:
        label = 'regression'
    else:
        label = 'classification'
    train_fingerprints = fingerprints_from_smiles_dataset(train_df, regression)
    test_fingerprints = fingerprints_from_smiles_dataset(test_df, regression)
    
    train_fingerprints.to_csv(f"/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/fingerprints/train_bace_fingerprints_{label}.csv",index=False)
    test_fingerprints.to_csv(f"/home/rajeckidoyle/Documents/Classification/BACE_Classification/model_benchmarks/fingerprints/test_bace_fingerprints_{label}.csv",index=False)


if __name__ == '__main__':
    generate_fingerprints(regression)
