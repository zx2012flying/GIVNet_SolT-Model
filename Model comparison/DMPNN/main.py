import numpy as np
from features import solute_descriptor_functions, solvent_descriptor_functions, calculate_descriptors
import pandas as pd
import warnings
import os
import argparse
import time
# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem
# torch imports
from torch.utils.data import DataLoader, Dataset
import torch
import dgl
from train import train
from data import MoleculeDataset, collate_fn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='DMPNN-MoE', help="The name of the current project: default: CIGIN")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | "
                                          "tanh-general", default='tm_distence')
parser.add_argument('--max_epochs', required=False, default=30, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, default=32, help="The batch size for training")

args = parser.parse_args()
project_name = args.name
interaction = args.interaction
max_epochs = int(args.max_epochs)
batch_size = int(args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.isdir("runs/run-" + str(project_name)):
    os.makedirs("./runs/run-" + str(project_name))
    os.makedirs("./runs/run-" + str(project_name) + "/models")

def main():
    for n in range(1, 11):
        train_df = pd.read_csv(f'train_dataset_fold_{n}_clean.csv')
        valid_df = pd.read_csv(f'test_dataset_fold_{n}_clean.csv')

        solute_descriptor_series = train_df['Solute_SMILES'].apply(
            lambda x: calculate_descriptors(x, solute_descriptor_functions))
        solvent_descriptor_series = train_df['Solvent_SMILES'].apply(
            lambda x: calculate_descriptors(x, solvent_descriptor_functions))
        df_solute_descriptors = pd.DataFrame(solute_descriptor_series.tolist()).add_prefix('solute_')
        df_solvent_descriptors = pd.DataFrame(solvent_descriptor_series.tolist()).add_prefix('solvent_')

        extra_features_df_train = pd.concat([train_df[['temperature/K']], df_solute_descriptors, df_solvent_descriptors], axis=1)
        extra_features_train = extra_features_df_train.values.astype(np.float32)

        solute_smiles_list_train = train_df['Solute_SMILES'].tolist()
        solvent_smiles_list_train = train_df['Solvent_SMILES'].tolist()


        solute_descriptor_series = valid_df['Solute_SMILES'].apply(
            lambda x: calculate_descriptors(x, solute_descriptor_functions))
        solvent_descriptor_series = valid_df['Solvent_SMILES'].apply(
            lambda x: calculate_descriptors(x, solvent_descriptor_functions))
        df_solute_descriptors = pd.DataFrame(solute_descriptor_series.tolist()).add_prefix('solute_')
        df_solvent_descriptors = pd.DataFrame(solvent_descriptor_series.tolist()).add_prefix('solvent_')

        extra_features_df_test = pd.concat([valid_df[['temperature/K']], df_solute_descriptors, df_solvent_descriptors], axis=1)
        extra_features_test = extra_features_df_test.values.astype(np.float32)

        solute_smiles_list_test = valid_df['Solute_SMILES'].tolist()
        solvent_smiles_list_test = valid_df['Solvent_SMILES'].tolist()


        train_labels = train_df['LogS(mol/L)'].values.astype(np.float32)
        valid_labels = valid_df['LogS(mol/L)'].values.astype(np.float32)

        train_dataset = MoleculeDataset(solute_smiles_list_train, solvent_smiles_list_train, extra_features_train, train_labels)
        valid_dataset = MoleculeDataset(solute_smiles_list_test, solvent_smiles_list_test, extra_features_test, valid_labels)

        train_loader = DataLoader(train_dataset, collate_fn= collate_fn, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, collate_fn= collate_fn, batch_size=128)

        train(max_epochs, train_loader, valid_loader, project_name, n)

if __name__ == '__main__':
    main()
