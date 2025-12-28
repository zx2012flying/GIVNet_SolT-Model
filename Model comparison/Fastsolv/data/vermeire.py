# vermeire.py
#
# Usage: python vermeire.py
#
# Requires: pandas, numpy, fastprop, rdkit
#
# Calculate molecular features needed for fastprop modeling
#
# Start by downloading the dataset from Zenodo:
# https://zenodo.org/records/5970538/files/SolProp_v1.2.zip?download=1
# and decompressing it in this directory.
#
# Running this will then csv files for fastprop training.
from pathlib import Path

import pandas as pd

from utils import get_descs, DROP_WATER

def main():
    # load the two datafiles and con
    #
    #
    #
    #
    #
    # catenate them
    # _src_dir: str = Path("SolProp_v1.2/Data")
    # room_T_data: pd.DataFrame = pd.read_csv(_src_dir / "CombiSolu-Exp-HighT.csv")
    # high_T_data: pd.DataFrame = pd.read_csv(_src_dir / "CombiSolu-Exp.csv")
    # all_data: pd.DataFrame = pd.concat((room_T_data, high_T_data))
    all_data = pd.read_csv("train_dataset_fold_1_clean.csv")
    # drop those missing the solubility
    all_data: pd.DataFrame = all_data[all_data["LogS(mol/L)"].notna()].reset_index()
    # rename columns
    all_data = all_data.rename(columns={"LogS(mol/L)": "logS"})

    fastprop_data: pd.DataFrame = get_descs(all_data)

    _dest = Path("vermeire")
    if not Path.exists(_dest):
        Path.mkdir(_dest)

    if DROP_WATER:
        fastprop_data = fastprop_data[~fastprop_data["solvent_smiles"].eq("O")].reset_index(drop=True)

    fastprop_data[["solute_smiles", "solvent_smiles", "temperature", "logS"]].to_csv(
        _dest / f"solprop_chemprop{'_nonaq' if DROP_WATER else ''}.csv",
    )
    fastprop_data.to_csv(_dest / f"traindata{'_nonaq' if DROP_WATER else ''}.csv")
if __name__ == "__main__":
    main()