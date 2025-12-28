import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from utils import one_of_k_encoding_unk, one_of_k_encoding
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]  # 映射到 "Unknown"
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom, stereo, features, explicit_H=False):

    possible_atoms = ['H', 'B', 'C', 'N', 'O', 'F', 'Na', 'Si', 'P', 'S', 'Cl', 'Ge', 'Se', 'Br', 'Sn', 'Te', 'I']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)  # 17
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3])  # 4
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])  # 2
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])  # 7
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])  # 3
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])  # 5
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:

        atom_features += [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]

    atom_features += one_of_k_encoding(atom.GetIsAromatic(), [0, 1])
    atom_features += one_of_k_encoding(atom.IsInRing(), [0, 1])

    return np.array(atom_features)


def get_bond_features(bond):

    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)


def get_graph_from_smile(molecule):

    features = rdDesc.GetFeatureInvariants(molecule)

    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):

        atom_i = molecule.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(bonds, dtype=torch.long).T
    edge_feats = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=atom_feats, edge_index=edge_index, edge_attr=edge_feats, num_nodes=molecule.GetNumAtoms())
