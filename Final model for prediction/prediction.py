import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch
sys.path.insert(0, './scripts/')
from molecular_graph import get_graph_from_smile
from model import PISGNN
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
import numpy as np
from rdkit import Chem
import random
import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(0)


def create_batch_mask(samples):
    graph0, graph1 = samples[0], samples[1]

    batch0 = graph0.batch
    num_nodes0 = batch0.size(0)
    index0 = torch.stack([batch0, torch.arange(num_nodes0, device=batch0.device)], dim=0)
    mask0 = torch.sparse_coo_tensor(
        indices=index0,
        values=torch.ones(num_nodes0, device=batch0.device),
        size=(batch0.max() + 1, num_nodes0)
    )

    batch1 = graph1.batch
    num_nodes1 = batch1.size(0)
    index1 = torch.stack([batch1, torch.arange(num_nodes1, device=batch1.device)], dim=0)
    mask1 = torch.sparse_coo_tensor(
        indices=index1,
        values=torch.ones(num_nodes1, device=batch1.device),
        size=(batch1.max() + 1, num_nodes1)
    )

    return mask0, mask1

device = "cuda" if torch.cuda.is_available() else "cpu"
model=PISGNN().to(device)
model.load_state_dict(torch.load('weights/best_model_2.tar', map_location=device))
model.eval()


results = []
input_excel = "all_pairs.csv"
df = pd.read_csv(input_excel)
batch_size = 16
for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i + batch_size]
    with torch.no_grad():
        for idx, row in batch_df.iterrows():
            try:
                solute = row['Solute_SMILES']
                solvent = row['Solvent_SMILES']
                tm = row['temperature/K(conver)']
                # LogS = row['LogS(mol/L)']
                # Tm = torch.tensor(tm, dtype=torch.float32).to(device)
                Tm = torch.tensor(tm).to(device).float()

                mol_solute = Chem.MolFromSmiles(solute)
                mol_solute = Chem.AddHs(mol_solute)
                solute_graph = get_graph_from_smile(mol_solute)

                mol_solvent = Chem.MolFromSmiles(solvent)
                mol_solvent = Chem.AddHs(mol_solvent)
                solvent_graph = get_graph_from_smile(mol_solvent)


                solute_graph.batch = torch.zeros(solute_graph.num_nodes, dtype=torch.long).to(device)
                solvent_graph.batch = torch.zeros(solvent_graph.num_nodes, dtype=torch.long).to(device)

                masks = create_batch_mask([solute_graph, solvent_graph])

                solute_len = masks[0].to(device)
                solvent_len = masks[1].to(device)


                input_data = [
                    solute_graph.to(device),
                    solvent_graph.to(device),
                    solute_len,
                    solvent_len,
                    Tm
                ]

                LogS_pred, A, B, _, _, _ =  model(input_data)

                print("Predicted Solubility: ",str(LogS_pred.item()))
                results.append({
                    'solute': solute,
                    'solvent': solvent,
                    'temperature': Tm,
                    # 'LogS': LogS,
                    'LogS_pred': LogS_pred.item(),
                    'A_pred': A.item(),
                    'B_pred': B.item()})

                del mol_solute, mol_solvent, solute_graph, solvent_graph
                del masks, solute_len, solvent_len, Tm
                del LogS_pred, A, B
                if device == "cuda" and idx % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Failed SMILES: {solute} or {solvent}")
                print(f"Error processing row {idx}: {e}")

output_excel = "output.xlsx"
result_df = pd.DataFrame(results)
result_df.to_excel(output_excel, index=False)

