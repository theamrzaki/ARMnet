from csv import writer
import math
import time
import warnings
from datetime import datetime

from matplotlib.pylab import beta
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch.optim as optim
from sknetwork.ranking import PageRank
from torch.optim import lr_scheduler
from RCAEval.io.time_series import preprocess, drop_constant
from utils import compute_kl_divergence
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from RCAEval.io.time_series import preprocess

from RCAEval.e2e.models import (
        iTransformer, TimeMixerpp, Dlinear, Fits
    )
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

_EPS = 1e-10

# ========================================
# VAE utility functions
# ========================================
def get_triu_indices(num_nodes):  # NOTE
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):  # NOTE
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):  # NOTE
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):  # NOTE
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.0
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):  # NOTE
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.0
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def kl_gaussian_sem(preds):  # NOTE
    mu = preds
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (preds.size(0))) * 0.5


def nll_gaussian(preds, target, variance, add_const=False):  # NOTE
    mean1 = preds
    mean2 = target
    neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2.0 * np.exp(2.0 * variance))
    if add_const:
        const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))


def preprocess_adj_new_old(adj):  # NOTE
    if CONFIG.cuda:
        adj_normalized = torch.eye(adj.shape[0]).double().cuda() - (adj.transpose(0, 1))
    else:
        adj_normalized = torch.eye(adj.shape[0]).double() - (adj.transpose(0, 1))
    return adj_normalized


def preprocess_adj_new1_old(adj):  # NOTE
    if CONFIG.cuda:
        adj_normalized = torch.inverse(
            torch.eye(adj.shape[0]).double().cuda() - adj.transpose(0, 1)
        )
    else:
        adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double() - adj.transpose(0, 1))
    return adj_normalized




def preprocess_adj_new(adj):
    """Compute I - A^T"""
    device = adj.device
    I = torch.eye(adj.shape[0], device=device, dtype=adj.dtype)  # Use same dtype as input
    return I - adj.transpose(0, 1)

def preprocess_adj_new1(adj):
    """Compute (I - A^T)^(-1)"""
    device = adj.device
    I = torch.eye(adj.shape[0], device=device, dtype=adj.dtype)
    return torch.linalg.inv(I - adj.transpose(0, 1))

def isnan(x):  # NOTE
    return x != x


def matrix_poly(matrix, d):  # NOTE
    if CONFIG.cuda:
        x = torch.eye(d).double().cuda() + torch.div(matrix, d)
    else:
        x = torch.eye(d).double() + torch.div(matrix, d)
    return torch.matrix_power(x, d)


# matrix loss: makes sure at least A connected to another parents for child
def A_connect_loss(A, tol, z):  # NOTE
    d = A.size()[0]
    loss = 0
    for i in range(d):
        loss += 2 * tol - torch.sum(torch.abs(A[:, i])) - torch.sum(torch.abs(A[i, :])) + z * z
    return loss


# element loss: make sure each A_ij > 0
def A_positive_loss(A, z_positive):  # NOTE
    result = -A + z_positive * z_positive
    loss = torch.sum(result)

    return loss


class CONFIG:  # NOTE
    """Dataclass with app parameters"""

    def __init__(self):
        pass

    # You must change this to the filename you wish to use as input data!
    # data_filename = "alarm.csv"

    # Epochs
    epochs = 10

    # Batch size (note: should be divisible by sample size, otherwise throw an error)
    batch_size = 50

    # Learning rate (baseline rate = 1e-3)
    lr = 1e-3

    x_dims = 1
    z_dims = 1
    # data_variable_size = 12
    optimizer = "Adam"
    graph_threshold = 0.3
    tau_A = 0.0
    lambda_A = 0.0
    c_A = 1
    use_A_connect_loss = 0
    use_A_positiver_loss = 0
    # no_cuda = True
    encoder_hidden = 128
    decoder_hidden = 128
    temp = 0.5
    k_max_iter = 1
    encoder = "mlp"
    decoder = "mlp"
    no_factor = False
    encoder_dropout = 0.0
    decoder_dropout = (0.0,)
    h_tol = 1e-8
    lr_decay = 200
    gamma = 1.0
    prior = False


CONFIG.cuda = torch.cuda.is_available()
CONFIG.factor = not CONFIG.no_factor



import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from statsmodels.tsa.stattools import grangercausalitytests
from utils import (pot)

# =========================
# POT algorithm wrapper
# =========================
def compute_pot_scores(errors, risk=1e-2, init_level=0.98, num_candidates=100, epsilon=1e-8):
    """
    Compute POT-based anomaly scores from reconstruction errors.
    Includes safeguards for tiny values and fallback to quantile.
    """
    errors = np.asarray(errors).ravel()

    # Rescale if values are too small
    if errors.max() < 1e-6:
        errors = errors * 1e6

    try:
        # POT thresholding
        z, t = pot(errors,
                   risk=risk,
                   init_level=init_level,
                   num_candidates=num_candidates,
                   epsilon=epsilon)
    except Exception as e:
        # Fallback: simple quantile threshold
        z = np.quantile(errors, 1 - risk)
        t = np.where(errors > z)[0]

    # Normalize scores
    scores = np.maximum(0, errors - z)
    if scores.max() > 0:
        scores = scores / scores.max()

    return scores, z, t


def _sparsity_loss(coeffs, alpha):
    norm2 = torch.mean(torch.norm(coeffs, dim=1, p=2))
    norm1 = torch.mean(torch.norm(coeffs, dim=1, p=1))
    return (1 - alpha) * norm2 + alpha * norm1


# =========================
# Main causal RCA function
# =========================
def RMDnet(data, inject_time=None, dataset=None, with_bg=False, ensemble_method="static", with_baro_pre=False, with_baro_post=False, model_class=None, model_config=None, scalar_type=None, **kwargs):


    print(f"with_baro_post={with_baro_post}")
    print(f"with_baro_pretrain={with_baro_pre}")
    print(f"scalar_type={scalar_type}")
    if type(data) == dict: # multimodal
        metric = data["metric"]
        logts = data["logts"]
        # traces_err = data["tracets_err"]
        # traces_lat = data["tracets_lat"]

        # === metric ===
        metric = metric.iloc[::15, :]

        # == metric ==
        normal_metric = metric[metric["time"] < inject_time]
        anomal_metric = metric[metric["time"] >= inject_time]
        normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
        anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
        intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
        normal_metric = normal_metric[intersect]
        anomal_metric = anomal_metric[intersect]
        metric = pd.concat([normal_metric, anomal_metric], axis=0, ignore_index=True)
        data = metric
        print(f"{normal_metric.shape=}")
        print(f"{anomal_metric.shape=}")
        print(f"{metric.shape=}")
        print("with metric", data.shape)

        # == logts ==
        logts = drop_constant(logts)
        normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
        anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])
        log = pd.concat([normal_logts, anomal_logts], axis=0, ignore_index=True)
        data = pd.concat([data, log], axis=1)
        print(f"{normal_logts.shape=}")
        print(f"{anomal_logts.shape=}")
        print(f"{log.shape=}")
        print("with log", data.shape)
        data.to_csv("debug_withlog.csv", index=False)

        # print(f"{normalize=} {addup=}")

        # # == traces_err ==
        # if dataset == "mm-tt" or dataset == "mm-ob":
        #     traces_err = traces_err.fillna(method='ffill')
        #     traces_err = traces_err.fillna(0)
        #     traces_err = drop_constant(traces_err)

        #     normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        #     anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])
        #     trace = pd.concat([normal_traces_err, anomal_traces_err], axis=0, ignore_index=True)
        #     data = pd.concat([data, trace], axis=1)
        #     print(f"{normal_traces_err.shape=}")
        #     print(f"{anomal_traces_err.shape=}")
        #     print(f"{trace.shape=}")
        #     print("with traces_err", data.shape)
        # 
        #  # == traces_lat ==
        # if dataset == "mm-tt" or dataset == "mm-ob":
        #     traces_lat = traces_lat.fillna(method='ffill')
        #     traces_lat = traces_lat.fillna(0)
        #     traces_lat = drop_constant(traces_lat)
        #     normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        #     anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])
        #     trace = pd.concat([normal_traces_lat, anomal_traces_lat], axis=0, ignore_index=True)
        #     data = pd.concat([data, trace], axis=1)
        #     print(f"{normal_traces_lat.shape=}")
        #     print(f"{anomal_traces_lat.shape=}")
        #     print(f"{trace.shape=}")
        #     print("with traces_lat", data.shape)

        # dump to debug.csv
        # data.to_csv("debug.csv", index=False)
        # drop duplicated columns
        data = data.loc[:, ~data.columns.duplicated()]
        data = data.fillna(0)

    else:
        data = preprocess(
            data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
        )

    data /= data.max()

    data_sample_size = data.shape[0]
    data_variable_size = data.shape[1]

    node_names = data.columns.to_list()

    # graph construction, get the adj
    train_data = data

    # Generate off-diagonal interaction graph
    off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)

    # add adjacency matrix A
    num_nodes = data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))

    # =========================
    # Inside RMDnet, before training
    # =========================

    if False:
        # ----------------------------
        # Compute BARO z-scores per variable
        # ----------------------------
        split_idx = len(train_data) // 2
        normal_df = train_data.iloc[:split_idx]
        anomal_df = train_data.iloc[split_idx:]

        baro_scores_per_var = []
        for col in node_names:
            a = normal_df[col].to_numpy()
            b = anomal_df[col].to_numpy()
            scaler = RobustScaler().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            baro_scores_per_var.append(zscores)

        baro_scores_per_var = np.stack(baro_scores_per_var, axis=1)  # shape: (num_samples//2, num_vars)

        # Pad/truncate to match train_data size
        if baro_scores_per_var.shape[0] < train_data.shape[0]:
            pad = np.zeros((train_data.shape[0] - baro_scores_per_var.shape[0], train_data.shape[1]))
            baro_scores_per_var = np.vstack([baro_scores_per_var, pad])
        else:
            baro_scores_per_var = baro_scores_per_var[:train_data.shape[0]]

        # ----------------------------
        # Normalize BARO features to match scale of original data
        # ----------------------------
        baro_scores_per_var = (baro_scores_per_var - baro_scores_per_var.mean(axis=0)) / (
            baro_scores_per_var.std(axis=0) + 1e-8
        )

        # ----------------------------
        # Optional weighting factor to prevent domination
        # ----------------------------
        BARO_WEIGHT = getattr(CONFIG, "pre_baro_weight", 0.5)  # tune between 0.0 and 1.0
        baro_scores_per_var *= BARO_WEIGHT

        # ----------------------------
        # Add BARO as additional features
        # ----------------------------
        baro_df = pd.DataFrame(baro_scores_per_var, columns=[f"baro_{n}" for n in node_names])
        train_data = pd.concat([train_data.reset_index(drop=True), baro_df], axis=1)

        # Update encoder dimensions
        data_variable_size = train_data.shape[1]
        node_names = train_data.columns.to_list()


    func = globals()[model_class]
    model_config.enc_in = data_variable_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = func.Model(model_config).to(device)
    encoder = encoder.float()  # ensure FP32 (not double)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model on: {next(encoder.parameters()).device}")
    optimizer = optim.Adam(encoder.parameters(), lr=CONFIG.lr)

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG.lr_decay, gamma=CONFIG.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    # =========================
    # Triangular indices on GPU
    # =========================
    triu_indices = get_triu_offdiag_indices(data_variable_size).to(device)
    tril_indices = get_tril_offdiag_indices(data_variable_size).to(device)

    if CONFIG.cuda:
        encoder.cuda()
        #decoder.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    # compute constraint h(A) value
    def _h_A(A, m):
        expm_A = matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    prox_plus = torch.nn.Threshold(0.0, 0.0)

    def stau(w, tau):
        w1 = prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def update_optimizer(optimizer, original_lr, c_A):
        """related LR to c_A, whenever c_A gets big, reduce LR proportionally"""
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group["lr"] = lr

        return optimizer, lr

    # ----------------------------
    # Precompute diffusion schedule once (outside training loop)
    # ----------------------------
    T = 100  # max diffusion steps
    device = torch.device("cuda" if CONFIG.cuda else "cpu")

    betas = torch.linspace(1e-4, 0.02, T, device=device)   # put directly on correct device
    alpha = 1.0 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)   # [T]


    timing_stats = {"enc": [], "dec": [], "loss": [], "back": []}

    def compute_baro_for_batch(batch_tensor, split_ratio=0.5):
        """
        Compute BARO-style z-scores for a batch tensor.
        batch_tensor: (B, seq_len, num_vars)
        Returns: same shape tensor with BARO scores
        """
        B, seq_len, num_vars = batch_tensor.shape
        baro_tensor = torch.zeros_like(batch_tensor)

        split_idx = int(seq_len * split_ratio)
        normal_part = batch_tensor[:, :split_idx, :]
        anomal_part = batch_tensor[:, split_idx:, :]

        for i in range(num_vars):
            normal_i = normal_part[:, :, i].reshape(-1, 1).cpu().numpy()
            anomal_i = anomal_part[:, :, i].reshape(-1, 1).cpu().numpy()
            if len(normal_i) == 0:  # safety
                continue
            scaler = RobustScaler().fit(normal_i)
            zscores = scaler.transform(anomal_i)
            # pad back to original seq_len
            baro_values = np.concatenate([np.zeros(split_idx), zscores.flatten()])
            baro_tensor[:, :, i] = torch.tensor(baro_values[:seq_len], device=batch_tensor.device)

        return baro_tensor

    # ===================================
    # training: simplified AE (reconstruction only)
    # ===================================
    def train(epoch, optimizer, batch_size=64):
        encoder.train()
        #decoder.train()


        # Create input and next-step pairs
        x_data = train_data[:-1]      # all except last
        next_data = train_data[1:]    # all except first

        tensor_x = torch.tensor(x_data.to_numpy(), dtype=torch.float32).to(device)
        tensor_next = torch.tensor(next_data.to_numpy(), dtype=torch.float32).to(device)

        #dataset = torch.utils.data.TensorDataset(tensor_x, tensor_next)
        #loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        
        seq_len = 12
        batch_size = 64

        # sliding window
        x_seq = []
        next_seq = []

        for i in range(len(tensor_x) - seq_len):
            x_seq.append(tensor_x[i:i+seq_len])
            next_seq.append(tensor_next[i:i+seq_len])

        x_seq = torch.stack(x_seq)       # shape: (num_windows, seq_len, vars)
        next_seq = torch.stack(next_seq)

        dataset = torch.utils.data.TensorDataset(x_seq, next_seq)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        mse_loss = nn.MSELoss()
        total_loss = 0.0
        total_mse = 0.0
        num_samples = 0
        for batch in loader:
            # forward
            x, next = batch
            x = x.to(device).float()
            next = next.to(device).float()
            # x make it 3d with time lag 1
            # batch, time lag = 1, num_vars
            #x = x.view(x.size(0), seq_len, x.size(1)).float()
            #next = next.view(next.size(0), seq_len, next.size(1))
            x = x.float().to(device)
            next = next.float().to(device)

            #preds, coeff = encoder(x)
            #z = preds - next.squeeze(1)
            #x_recon = decoder(z)
            x_recon = encoder(x)
            coeff = None

            # reconstruction accuracy loss
            #loss_nll = nll_gaussian(x_recon, x.squeeze(1), 0.0)
            loss_mse = mse_loss(x_recon.squeeze(1), next.squeeze(1))
            #loss_encoder_coeffs = _sparsity_loss(coeff, 0.5)# encoder_alpha = 0.5
            #kl_div = compute_kl_divergence(z, device)
            # KL loss
            #loss_kl = kl_gaussian_sem(z)
            BARO_WEIGHT = 0.05  # start small
            if with_baro_pre:
                

                baro_batch = compute_baro_for_batch(x)
                baro_batch = torch.clamp(baro_batch, -10.0, 10.0).float().to(device)

                loss_mse = mse_loss(x_recon, next)
                loss_baro = mse_loss(x_recon, baro_batch)


            else:
                loss_baro = 0.0
            # ELBO loss:
            loss = loss_mse + BARO_WEIGHT * loss_baro

            #loss = loss_kl + loss_mse #+ 0.1 * kl_div
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(x)
            total_mse += loss.item() * len(x)
            num_samples += len(x)

        avg_loss = total_loss / num_samples
        avg_mse = total_mse / num_samples

        print(f"Epoch {epoch:03d} | Loss={avg_loss:.6f}")

        return avg_loss, avg_mse

    # ===================================
    # main
    # ===================================

    # gamma = 0.5
    gamma = 0.25
    eta = 10

    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = CONFIG.c_A
    lambda_A = CONFIG.lambda_A
    h_A_new = torch.tensor(1.0)
    h_tol = CONFIG.h_tol
    k_max_iter = int(CONFIG.k_max_iter)
    h_A_old = np.inf

    E_loss = []
    N_loss = []
    M_loss = []
    start_time = time.time()
    # name of experiment for TensorBoard logging
    
    exp_name = "causalrca_experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"./runs/{exp_name}")
    try:
        for step_k in range(k_max_iter):
            # print(step_k)
            #while c_A < 1e20:
            for epoch in range(CONFIG.epochs):
                # print(epoch)
                # only log first epoch
                avg_loss, avg_mse = train(
                    epoch, optimizer
                )
                # ----------------- TensorBoard Logging -----------------
                if writer is not None:
                    writer.add_scalar("loss/ELBO", avg_loss, step_k * CONFIG.epochs + epoch)
                    writer.add_scalar("loss/MSE", avg_mse, step_k * CONFIG.epochs + epoch)
                    writer.add_scalar("h_A", h_A_new.item(), step_k * CONFIG.epochs + epoch)
                    writer.add_scalar("c_A", c_A, step_k * CONFIG.epochs + epoch)
                    writer.add_scalar("lambda_A", lambda_A, step_k * CONFIG.epochs + epoch)
                # -------------------------------------------------------
                    

                # print(f"{ELBO_loss=} {NLL_loss=} {MSE_loss=}")
                E_loss.append(avg_loss)
                N_loss.append(avg_mse)
                M_loss.append(avg_mse)
                if avg_loss < best_ELBO_loss:
                    best_ELBO_loss = avg_loss
                    best_epoch = epoch

                if avg_mse < best_NLL_loss:
                    best_NLL_loss = avg_mse
                    best_epoch = epoch

                if avg_mse < best_MSE_loss:
                    best_MSE_loss = avg_mse
                    best_epoch = epoch

            # print("Optimization Finished!")
            # print("Best Epoch: {:04d}".format(best_epoch))
            if avg_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break
            # after finishing this step_k
        print(f"[step_k={step_k}] "
            f"avg encoder={np.mean(timing_stats['enc']):.4f}s, "
            f"avg decoder={np.mean(timing_stats['dec']):.4f}s, "
            f"avg loss={np.mean(timing_stats['loss']):.4f}s, "
            f"avg backward={np.mean(timing_stats['back']):.4f}s"
        )
        

    except KeyboardInterrupt:
        print("Done!")

    writer.close()
    # just build on my code to apply POT for ranks 
    # ===================================
    # After training → reconstruction & POT scoring
    # ===================================
    #decoder.eval()
    seq_len = 12
    pred_len = 1
    batch_size = 64
    with torch.no_grad():
        encoder.eval()
        data_np = data.to_numpy()
        num_samples, num_vars = data_np.shape

        # Build sliding windows for sequence input
        sequences = []
        for i in range(num_samples - seq_len + 1):
            sequences.append(data_np[i:i+seq_len])
        sequences = np.stack(sequences, axis=0)  # shape: (num_sequences, seq_len, num_vars)

        sequences_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        num_sequences = sequences_tensor.size(0)

        preds_list = []
        recon_list = []

        with torch.no_grad():
            encoder.eval()
            for start in range(0, num_sequences, batch_size):
                end = start + batch_size
                x_batch = sequences_tensor[start:end]  # (B, seq_len, num_vars)

                recon_batch = encoder(x_batch)  # forecast: (B, pred_len, num_vars), recon: (B, seq_len+pred_len, num_vars)

                recon_list.append(recon_batch.cpu()[:, :seq_len, :])  # only keep reconstruction for input sequence

            sequences_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)


        recon_tensor = torch.cat(recon_list, dim=0)

        recon_np = recon_tensor.numpy()  # (num_sequences, seq_len, num_vars)

        # Compute residuals and z-scores
        residual_np = recon_np - sequences  # (num_sequences, seq_len, num_vars)
        res_mean = residual_np.mean(axis=0)
        res_std = residual_np.std(axis=0) + 1e-8
        residual_z = -(residual_np - res_mean) / res_std

        # === POT threshold per variable (latent / encoder–decoder) ===
        # Suppose residual_z has shape: (num_sequences, seq_len, num_vars)
        # Collapse sequence dimension (e.g., mean or flatten)
        res_z_flat = residual_z.reshape(-1, data_variable_size)  # shape: (num_sequences*seq_len, num_vars)

        scores = []
        for i in range(data_variable_size):
            pot_val, _, _ = compute_pot_scores(
                res_z_flat[:, i],  # use flattened residual z-score for variable i
                risk=getattr(CONFIG, "pot_risk", 1e-2),
                init_level=getattr(CONFIG, "pot_init_level", 0.98),
                num_candidates=getattr(CONFIG, "pot_num_candidates", 10),
                epsilon=getattr(CONFIG, "pot_epsilon", 1e-8),
            )
            scores.append(pot_val)

        scores = np.array(scores)
        ed_scores = np.array([val.mean() for val in scores])  # mean anomaly fraction per variable

        if with_baro_post:
            # === BARO-style statistical deviation per variable ===
            # Split into "normal" and "anomalous" parts — you can use your inject_time or anomaly split logic
            split_idx = len(train_data) // 2  # fallback simple split
            normal_df = train_data.iloc[:split_idx]
            anomal_df = train_data.iloc[split_idx:]

            baro_scores = []
            for col in node_names:
                a = normal_df[col].to_numpy()
                b = anomal_df[col].to_numpy()
                print(scalar_type)
                if scalar_type == "Robust":#baro
                    scaler = RobustScaler().fit(a.reshape(-1, 1))
                elif scalar_type == "Standard":#nsigma
                    scaler = StandardScaler().fit(a.reshape(-1, 1))
                zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
                baro_scores.append(np.max(zscores))  # BARO = max deviation
            baro_scores = np.array(baro_scores)

            # === Combine both (hybrid fusion) ===
            if ensemble_method == "static":
                alpha = 0.6  # weight for encoder–decoder signal
                hybrid_scores = alpha * ed_scores + (1 - alpha) * baro_scores

            elif ensemble_method == "dynamic":#(no difference to the static)
                # Example: inverse of standard deviation as confidence
                min_len = min(len(normal_df), len(anomal_df))
                diff = anomal_df.iloc[:min_len].to_numpy() - normal_df.iloc[:min_len].to_numpy()
                baro_conf = 1 / (np.std(diff, axis=0) + 1e-8)

                ed_conf = 1 / (np.std(res_z_flat, axis=0) + 1e-8)

                # Normalize to get weights
                total = ed_conf + baro_conf
                alpha_dyn = ed_conf / total

                hybrid_scores = alpha_dyn * ed_scores + (1 - alpha_dyn) * baro_scores
            elif ensemble_method == "rank":
                ed_ranks = np.argsort(np.argsort(-ed_scores))  # descending
                baro_ranks = np.argsort(np.argsort(-baro_scores))

                # Dynamic weighting could depend on agreement
                agreement = np.abs(ed_ranks - baro_ranks)
                alpha_dyn = 1 - (agreement / agreement.max())

                hybrid_scores = alpha_dyn * ed_scores + (1 - alpha_dyn) * baro_scores
            elif ensemble_method == "sigmoid":#(no difference to the static)
                def sigmoid(x, beta=1):
                    return 1 / (1 + np.exp(-beta * x))

                ed_prob = sigmoid(ed_scores)
                baro_prob = sigmoid(baro_scores)

                # Weighted combination based on max confidence
                alpha_dyn = ed_prob / (ed_prob + baro_prob + 1e-8)
                hybrid_scores = alpha_dyn * ed_scores + (1 - alpha_dyn) * baro_scores

        else:
            baro_scores = np.zeros_like(ed_scores)
            hybrid_scores = ed_scores
        # === Rank variables ===
        ranks = list(zip(node_names, hybrid_scores))
        ranks.sort(key=lambda x: x[1], reverse=True)
        ranks = [x[0] for x in ranks]

    # === Final return dict ===
    return {
        "scores": hybrid_scores.tolist(),
        "ranks": ranks,
        "node_names": node_names,
        "ed_scores": ed_scores.tolist(),
        "baro_scores": baro_scores.tolist(),
    }

if __name__ == "__main__":
    data = pd.read_csv("/home/luan/ws/cfm/tmp_data/cartservice_mem/1/data.csv")

    n = 30

    # read inject_time
    with open("/home/luan/ws/cfm/tmp_data/cartservice_mem/1/inject_time.txt", "r") as f:
        inject_time = f.read()
    inject_time = int(inject_time)
    normal_df = data[data["time"] <= inject_time].tail(n)
    anomalous_df = data[data["time"] > inject_time].head(n)
    data = pd.concat([normal_df, anomalous_df], ignore_index=True)

    output = causalrca(data, inject_time=None, dataset="ob")
    print(output)