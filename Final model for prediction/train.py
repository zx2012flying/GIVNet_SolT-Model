from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import os
import pandas as pd
import random
from utils import create_batch_mask, create_interv_mask
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import PISGNN
from torch import nn

loss_fn = torch.nn.SmoothL1Loss()
mae_loss_fn = torch.nn.L1Loss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_best_fold_results(y_true, y_pred, r2, rmse, mae,n):

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, ax_main = plt.subplots(figsize=(10, 8))

    min_val, max_val = min(y_true), max(y_true)
    offset = 0.5
    ideal_line = [min_val, max_val]
    upper_bound = [x + offset for x in ideal_line]
    lower_bound = [x - offset for x in ideal_line]

    ax_main.plot(ideal_line, ideal_line, color='red', linestyle='--', label='Ideal Line')
    ax_main.fill_between(ideal_line, lower_bound, upper_bound, color='#DD706E', alpha=0.1)

    within_bounds = ((y_pred <= (y_true + offset)) & (y_pred >= (y_true - offset)))
    within_bounds_ratio = sum(within_bounds) / len(y_true)*100

    scatter = ax_main.scatter(y_true, y_pred, alpha=0.3, label=f'R²: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\n% ±0.5: {within_bounds_ratio:.2f}')

    ax_main.set_xlabel('Exp. LogS')
    ax_main.set_ylabel('CIGIN LogS')
    ax_main.legend(loc='best')
    ax_main.grid(True)

    ax_hist_x = inset_axes(ax_main, width="100%", height="25%", loc='upper center',
                           bbox_to_anchor=(0, 0.25, 1, 1), bbox_transform=ax_main.transAxes, borderpad=0)

    ax_hist_y = inset_axes(ax_main, width="25%", height="100%", loc='center right',
                           bbox_to_anchor=(0.25, 0, 1, 1), bbox_transform=ax_main.transAxes, borderpad=0)

    bins = 30
    sns.histplot(y_true, bins=bins, kde=False, stat="density", color='#3A93C2', alpha=0.5, ax=ax_hist_x)
    sns.kdeplot(y_true, color='blue', alpha=0.7, ax=ax_hist_x)
    ax_hist_x.axis('off')

    sns.histplot(y=y_pred, bins=bins, kde=False, stat="density", color='#3A93C2', alpha=0.5, ax=ax_hist_y,
                 orientation='horizontal')
    sns.kdeplot(y=y_pred, color='blue', alpha=0.7, ax=ax_hist_y)
    ax_hist_y.axis('off')

    ax_hist_x.tick_params(axis="x", labelbottom=False)
    ax_hist_y.tick_params(axis="y", labelleft=False)

    ax_hist_x.set_xlim(ax_main.get_xlim())
    ax_hist_y.set_ylim(ax_main.get_ylim())

    save_path = os.path.join(os.getcwd(), f'pred_vs_actual_{n}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def get_metrics(model, data_loader):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    A_outputs = []
    B_outputs = []
    valid_A = []
    valid_B = []
    tq_loader = tqdm(data_loader)
    with torch.no_grad():
        for samples in tq_loader:

            masks = create_batch_mask(samples)
            tm = torch.tensor(samples[3]).to(device)
            pos, A_pred, B_pred, _, _, _ = model(
                [samples[0].to(device), samples[1].to(device), masks[0].to(device), masks[1].to(device),
                 tm])

            loss = loss_fn(pos, torch.tensor(samples[2]).to(device).float())
            mae_loss = mae_loss_fn(pos, torch.tensor(samples[2]).to(device).float())

            valid_outputs.extend(pos.cpu().detach().numpy().flatten().tolist())
            A_outputs.extend(A_pred.cpu().detach().numpy().flatten().tolist())
            B_outputs.extend(B_pred.cpu().detach().numpy().flatten().tolist())
            valid_loss.append(loss.cpu().detach().numpy())
            valid_mae_loss.append(mae_loss.cpu().detach().numpy())
            valid_labels.extend(np.array(samples[2]).flatten().tolist())
            valid_A.extend(np.array(samples[4]).flatten().tolist())
            valid_B.extend(np.array(samples[5]).flatten().tolist())
        loss = np.mean(np.hstack(valid_loss))
        mae_loss = np.mean(np.hstack(valid_mae_loss))
    return loss, mae_loss, np.array(valid_labels), np.array(valid_outputs), np.array(valid_A), np.array(A_outputs), \
        np.array(valid_B), np.array(B_outputs)


def contrastive_loss(rep_a, rep_b):

    batch_size, _ = rep_a.size()
    rep_a_abs = rep_a.norm(dim = 1)
    rep_b_abs = rep_b.norm(dim = 1)

    sim_matrix = torch.einsum('ik,jk->ij', rep_a, rep_b) / torch.einsum('i,j->ij', rep_a_abs, rep_b_abs)
    sim_matrix = torch.exp(sim_matrix / 0.2)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss

#
def train(max_epochs, train_loader, valid_loader, project_name, n):
    model =PISGNN().to(device)
    # model.load_state_dict(torch.load('best_model_1.tar'))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006744750458037359, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min')

    best_val_loss = 10000000
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start_time = time.time()
    for epoch in range(max_epochs):
        model.train()
        epoch_start_time = time.time()
        running_loss = []
        tq_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for samples in tq_loader:
            optimizer.zero_grad()


            tm = torch.tensor(samples[3], dtype=torch.float32).to(device).requires_grad_()

            masks = create_batch_mask(samples)
            pos, A_pred, B_pred,  neg, rand, inter_predict = model([samples[0].to(device), samples[1].to(device),
                 masks[0].to(device), masks[1].to(device),tm])

            loss = loss_fn(pos, torch.tensor(samples[2]).reshape(-1, 1).to(device).float())
            loss_neg = 0.9094997399481588 * loss_fn(neg, torch.zeros_like(pos).to(device).float())
            loss_inv = 0.14699298303427505 * loss_fn(rand, torch.tensor(samples[2]).reshape(-1, 1).to(device).float())

            eps = 1e-6
            safe_tm = torch.where(tm.abs() < eps, eps * torch.sign(tm), tm)
            with torch.enable_grad():
                safe_tm.requires_grad_(True)
                pred_value = A_pred + B_pred / safe_tm
                y_grad_hat = torch.autograd.grad(
                    outputs=pred_value,
                    inputs=safe_tm,
                    grad_outputs=torch.ones_like(pred_value),
                    create_graph=True,
                    retain_graph=True
                )[0]
            target_gradients = -B_pred / (safe_tm ** 2)

            scale_factor = 0.5062264539230115
            y_grad_loss = scale_factor * ((y_grad_hat - target_gradients).pow(2).nanmean())

            loss_A = 0.8684764305483391 * loss_fn(A_pred, torch.tensor(samples[4]).reshape(-1, 1).to(device).float())

            total_loss = loss + loss_inv + loss_neg
            total_loss += y_grad_loss
            total_loss += loss_A
            total_loss.backward()
            optimizer.step()
            running_loss.append(total_loss.cpu().detach())
            current_time = time.time() - start_time
            tq_loader.set_postfix({
                "total_loss": f"{total_loss.item():.4f}",
                "Train Time": f"{time.time() - epoch_start_time:.1f}s",
                "Total Time": f"{current_time:.1f}s"
            })

        model.eval()

        val_loss, mae_loss, y_true, y_pred, A_true, A_pred, B_true, B_pred = get_metrics(model, valid_loader)
        scheduler.step(val_loss)

        print("Epoch: " + str(epoch + 1) + "  train_loss " + str(np.mean(np.array(running_loss))) + " Val_loss " + str(
            val_loss) + " MAE Val_loss " + str(mae_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./runs/run-" + str(project_name) + f"/models/best_model_{n}.tar")

            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            plot_best_fold_results(y_true, y_pred, r2, rmse, mae ,n)

            A_true = np.array(A_true).flatten()
            A_pred = np.array(A_pred).flatten()
            B_true = np.array(B_true).flatten()
            B_pred = np.array(B_pred).flatten()
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()

            data = {
                'LogS' :y_true,
                'LogS_pred': y_pred,
                'A_true': A_true,
                'A_pred': A_pred,
                'B_true': B_true,
                'B_pred': B_pred
            }
            df = pd.DataFrame(data)
            df.to_csv(f'result_file_{n}.csv', index=False)

