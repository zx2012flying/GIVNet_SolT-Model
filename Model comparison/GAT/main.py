import pandas as pd
import warnings
import os
import argparse
# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from train import train
from molecular_graph import get_graph_from_smile
from utils import *
from torch_geometric.data import Data, Batch
import random
import time


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='PIS-GNN', help="The name of the current project: default: PIS-GNN")
parser.add_argument('--max_epochs', required=False, default = 150, help="The max number of epochs for training")
parser.add_argument('--train_batch_size', required=False, default = 32, help="The batch size for training")
parser.add_argument('--test_batch_size', required=False, default = 64, help="The batch size for test")

args = parser.parse_args()
project_name = args.name
max_epochs = int(args.max_epochs)
train_batch_size = int(args.train_batch_size)
test_batch_size = int(args.test_batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if not os.path.isdir("runs/run-" + str(project_name)):
    os.makedirs("./runs/run-" + str(project_name))
    os.makedirs("./runs/run-" + str(project_name) + "/models")

def preprocess_and_save_dataset(dataset, save_path):
    data_list = []
    for idx in range(len(dataset)):
        try:
            sample = dataset.__getitem__(idx)
            data_list.append(sample)
        except Exception as e:
            print(f"Skipped sample {idx}: {e}")

    torch.save(data_list, save_path)
    print(f"Saved dataset to {save_path}")



def collate(samples):
    solute_graphs, solvent_graphs, delta_g, tm, A, B= map(list, zip(*samples))

    solute_batch = Batch.from_data_list(solute_graphs)
    solvent_batch = Batch.from_data_list(solvent_graphs)

    return solute_batch, solvent_batch, delta_g, tm, A, B

class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        solute = self.dataset.loc[idx]['Solute_SMILES']
        mol_solute = Chem.MolFromSmiles(solute)
        mol_solute = Chem.AddHs(mol_solute)
        solute_graph = get_graph_from_smile(mol_solute)

        solvent = self.dataset.loc[idx]['Solvent_SMILES']
        mol_solvent = Chem.MolFromSmiles(solvent)
        mol_solvent = Chem.AddHs(mol_solvent)

        solvent_graph = get_graph_from_smile(mol_solvent)
        delta_g = self.dataset.loc[idx]['LogS(mol/L)']
        tm = self.dataset.loc[idx]['temperature/K(conver)']
        A = self.dataset.loc[idx]['A(conver)']
        B = self.dataset.loc[idx]['B(conver)']


        return [solute_graph, solvent_graph ,[delta_g], [tm], [A], [B]]


def main():

    total_start_time = time.time()
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    for n in range(4,6):
        print(f'Training on fold {n}...')
        train_df = pd.read_csv(f'data/train_dataset_fold_{n}_clean.csv')
        valid_df = pd.read_csv(f'data/test_dataset_fold_{n}_clean.csv')

        train_pt = os.path.join(processed_dir, f'train_data_{n}.pt')
        test_pt = os.path.join(processed_dir, f'test_data_{n}.pt')

        train_dataset = Dataclass(train_df)
        valid_dataset = Dataclass(valid_df)

        if not os.path.exists(train_pt):
            print("Preprocessing training data...")
            preprocess_and_save_dataset(train_dataset, train_pt)
        else:
            print(f"Train data already exists at {train_pt}")

        if not os.path.exists(test_pt):
            print("Preprocessing test data...")
            preprocess_and_save_dataset(valid_dataset, test_pt)
        else:
            print(f"Test data already exists at {test_pt}")

        train_data_list = torch.load(train_pt, weights_only=False)
        valid_data_list = torch.load(test_pt, weights_only=False)

        train_loader = DataLoader(train_data_list, batch_size= train_batch_size, shuffle=True, collate_fn=collate)
        test_loader = DataLoader(valid_data_list, batch_size= test_batch_size, collate_fn=collate)

        seed_everything()
        train(max_epochs, train_loader, test_loader, project_name,n)
    total_time = time.time() - total_start_time
    print(f"Total training time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()
