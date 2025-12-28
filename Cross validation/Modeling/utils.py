import numpy as np
import torch

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_len_matrix(len_list):
    len_list = np.array(len_list)
    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum + l] = 1
        len_matrix.append(curr)
        curr_sum += l
    return np.array(len_matrix)

def create_batch_mask(samples):
    batch0 = samples[0].batch.reshape(1, -1)
    index0 = torch.cat([batch0, torch.tensor(range(batch0.shape[1])).reshape(1, -1)])
    mask0 = torch.sparse_coo_tensor(index0, torch.ones(index0.shape[1]), size = (batch0.max() + 1, batch0.shape[1]))

    batch1 = samples[1].batch.reshape(1, -1)
    index1 = torch.cat([batch1, torch.tensor(range(batch1.shape[1])).reshape(1, -1)])
    mask1 = torch.sparse_coo_tensor(index1, torch.ones(index1.shape[1]), size = (batch1.max() + 1, batch1.shape[1]))

    return mask0, mask1


def create_interv_mask(rand_batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rand_batch = rand_batch.reshape(1, -1)
    index0 = torch.cat([rand_batch, torch.tensor(range(rand_batch.shape[1]), device=device).reshape(1, -1)])
    mask0 = torch.sparse_coo_tensor(index0, torch.ones(index0.shape[1], device=device), size = (rand_batch.max() + 1, rand_batch.shape[1]))

    return mask0
