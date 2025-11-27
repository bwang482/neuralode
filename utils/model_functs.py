
import os
import sys
import numpy as np
from tqdm.auto import tqdm
import sklearn.metrics as sk_metrics

import torch
import torch.nn as nn

from utils.data_functs import subsample_obs_data


def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device


def transfer_batch_to_device(batch_dict, device):
    """
    Moves all tensors in the batch dictionary to the specified device.
    """
    for key in batch_dict:
        if isinstance(batch_dict[key], torch.Tensor):
            batch_dict[key] = batch_dict[key].to(device)
    return batch_dict


def create_netw(n_input, n_output, n_layer = None, n_unit = None, nonlinear = nn.Tanh):

    layers = [nn.Linear(n_input, n_unit)]
    layers.append(nonlinear())   # input layer

    for ii in range(n_layer):
        layers.append(nn.Linear(n_unit, n_unit))
        layers.append(nonlinear())   # latent layers

    layers.append(nn.Linear(n_unit, n_output))   # output layer

    return nn.Sequential(*layers)   # a sequential container


def init_netw_weights(netw, std = 0.1):

    for mm in netw.modules():
        if isinstance(mm, nn.Linear):
            nn.init.normal_(mm.weight, mean = 0, std = std)
            nn.init.constant_(mm.bias, val = 0)


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):

    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr


def batch_postprocess(raw_batch_dict):
    # Create a new dict to avoid modifying the one from the dataloader directly
    batch_dict = {}
    # Filter obs_data based on non-missing time points across the batch
    if "obs_data" in raw_batch_dict and raw_batch_dict["obs_data"] is not None:
        # Check time points with any observation across batch and features
        non_missing_tp_obs = torch.sum(raw_batch_dict["obs_data"], (0, 2)) != 0.
        # Filter data, time steps, and mask
        batch_dict["obs_data"] = raw_batch_dict["obs_data"][:, non_missing_tp_obs]
        batch_dict["obs_tps"] = raw_batch_dict["obs_tps"][non_missing_tp_obs]
        if "obs_mask" in raw_batch_dict and raw_batch_dict["obs_mask"] is not None:
            batch_dict["obs_mask"] = raw_batch_dict["obs_mask"][:, non_missing_tp_obs]
        else:
            batch_dict["obs_mask"] = None # Ensure key exists if filtering happened
    else: # Handle cases where keys might be missing (optional, defensive)
        batch_dict["obs_data"] = None
        batch_dict["obs_tps"] = raw_batch_dict.get("obs_tps")
        batch_dict["obs_mask"] = None

    # Filter data_to_pred similarly
    if "data_to_pred" in raw_batch_dict and raw_batch_dict["data_to_pred"] is not None:
        non_missing_tp_pred = torch.sum(raw_batch_dict["data_to_pred"], (0, 2)) != 0.
        batch_dict["data_to_pred"] = raw_batch_dict["data_to_pred"][:, non_missing_tp_pred]
        batch_dict["tps_to_pred"] = raw_batch_dict["tps_to_pred"][non_missing_tp_pred]
        if "pred_mask" in raw_batch_dict and raw_batch_dict["pred_mask"] is not None:
            batch_dict["pred_mask"] = raw_batch_dict["pred_mask"][:, non_missing_tp_pred]
        else:
            batch_dict["pred_mask"] = None # Ensure key exists
    else: # Handle cases where keys might be missing
        batch_dict["data_to_pred"] = None
        batch_dict["tps_to_pred"] = raw_batch_dict.get("tps_to_pred")
        batch_dict["pred_mask"] = None

    # Copy other essential keys
    batch_dict["labels"] = raw_batch_dict.get("labels")
    batch_dict["mode"] = raw_batch_dict.get("mode")
    return batch_dict


def get_next_batch(dataloader):
    if hasattr(dataloader, '__next__'):
        data_dict = dataloader.__next__()
    else:
        data_dict = next(iter(dataloader))

    batch_dict = {"obs_data": None, "obs_tps": None, "obs_mask": None,
                  "data_to_pred": None, "tps_to_pred": None, "pred_mask": None, "labels": None}

    # remove time points that had no observations in this batch
    non_missing_tp = torch.sum(data_dict["obs_data"], (0,2)) != 0.   # n_subj x n_tp x n_dim
    batch_dict["obs_data"] = data_dict["obs_data"][:, non_missing_tp]
    batch_dict["obs_tps"] = data_dict["obs_tps"][non_missing_tp]

    if ("obs_mask" in data_dict) and (data_dict["obs_mask"] is not None):
        batch_dict["obs_mask"] = data_dict["obs_mask"][:, non_missing_tp]

    batch_dict["data_to_pred"] = data_dict["data_to_pred"]
    batch_dict["tps_to_pred"] = data_dict["tps_to_pred"]

    non_missing_tp = torch.sum(data_dict["data_to_pred"], (0,2)) != 0.
    batch_dict["data_to_pred"] = data_dict["data_to_pred"][:, non_missing_tp]
    batch_dict["tps_to_pred"] = data_dict["tps_to_pred"][non_missing_tp]

    if ("pred_mask" in data_dict) and (data_dict["pred_mask"] is not None):
        batch_dict["pred_mask"] = data_dict["pred_mask"][:, non_missing_tp]

    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        batch_dict["labels"] = data_dict["labels"]

    batch_dict["mode"] = data_dict["mode"]

    return batch_dict


def get_next_batch_distributed(dataloader):
    """
    Get next batch from dataloader, supporting both infinite generators and distributed dataloaders
    """
    # Check if dataloader is an iterator (infinite generator) or a DataLoader
    if hasattr(dataloader, '__next__'):
        # It's already an iterator - use directly
        data_dict = dataloader.__next__()
    else:
        # It's a regular DataLoader, not an iterator - get an iterator for it
        if not hasattr(dataloader, '_iterator') or dataloader._iterator is None:
            dataloader._iterator = iter(dataloader)
        
        try:
            data_dict = next(dataloader._iterator)
        except StopIteration:
            # Restart iterator if exhausted
            dataloader._iterator = iter(dataloader)
            data_dict = next(dataloader._iterator)

    batch_dict = {"obs_data": None, "obs_tps": None, "obs_mask": None,
            "data_to_pred": None, "tps_to_pred": None, "pred_mask": None, "labels": None}

    # remove time points that had no observations in this batch
    non_missing_tp = torch.sum(data_dict["obs_data"], (0,2)) != 0.   # n_subj x n_tp x n_dim
    batch_dict["obs_data"] = data_dict["obs_data"][:, non_missing_tp]
    batch_dict["obs_tps"] = data_dict["obs_tps"][non_missing_tp]

    if ("obs_mask" in data_dict) and (data_dict["obs_mask"] is not None):
        batch_dict["obs_mask"] = data_dict["obs_mask"][:, non_missing_tp]


    batch_dict["data_to_pred"] = data_dict["data_to_pred"]
    batch_dict["tps_to_pred"] = data_dict["tps_to_pred"]

    non_missing_tp = torch.sum(data_dict["data_to_pred"], (0,2)) != 0.
    batch_dict["data_to_pred"] = data_dict["data_to_pred"][:, non_missing_tp]
    batch_dict["tps_to_pred"] = data_dict["tps_to_pred"][non_missing_tp]

    if ("pred_mask" in data_dict) and (data_dict["pred_mask"] is not None):
        batch_dict["pred_mask"] = data_dict["pred_mask"][:, non_missing_tp]


    if ("labels" in data_dict) and (data_dict["labels"] is not None):
        batch_dict["labels"] = data_dict["labels"]

    batch_dict["mode"] = data_dict["mode"]

    return batch_dict


def split_last_dim(x):

    n_dim = x.size()[-1]
    n_dim = n_dim // 2

    if len(x.size()) == 3:
        splt = x[:,:,:n_dim], x[:,:,n_dim:]

    if len(x.size()) == 2:
        splt = x[:,:n_dim], x[:,n_dim:]

    return splt


def check_mask(data, mask):

    n_zero = torch.sum(mask == 0.).cpu().numpy()   # number of zeros
    n_one = torch.sum(mask == 1.).cpu().numpy()   # number of ones
    # move the tensor to cpu memory and convert a tensor into an numpy.ndarray object

    assert((n_zero + n_one) == np.prod(list(mask.size())))   # check that mask only contains zeros and ones

    assert(torch.sum(data[mask == 0.] != 0.) == 0)   # check that all masked out elements are zeros


def sample_standard_gaussian(mu, sigma):
    device = mu.device

    rnd = torch.randn(mu.size(), device=device).squeeze(-1)

    return rnd * sigma + mu


def compute_loss_all_batches(args, model, test_dataloader, n_traj, kl_coef, 
                             ce_weight=100, use_joint_loss=True, z_last=False):
    batch_dict = get_next_batch(test_dataloader)
    
    metrics = model.compute_all_losses(
         batch_dict, n_traj, kl_coef, ce_weight=ce_weight, 
         use_joint_loss=use_joint_loss, z_last=z_last
    )
    
    metrics["auc"] = 0.
    
    if args.classif:
        labels = batch_dict["labels"].reshape(-1)
        labels = labels.repeat(n_traj, 1)
        
        classif_pred = metrics["label_pred"].reshape(n_traj, -1)
        
        assert(torch.sum(labels) != 0.)
        
        all_labels = labels.reshape(-1).cpu()
        all_preds = classif_pred.reshape(-1).cpu()
        
        metrics["auc"] = sk_metrics.roc_auc_score(all_labels, all_preds)
    
    return metrics


def compute_loss_val_batches(args, model, test_dataloader, n_traj, kl_coef, 
                             ce_weight=100, use_joint_loss=True, z_last=False, device=None):
    total_samples = 0
    cumulative_metrics = {"loss": 0.0, "likelihood": 0.0, "mse": 0.0, 
                          "ce_loss": 0.0, "kl_div": 0.0, "fp_std": 0.0}
    all_labels = []
    all_preds = []
    
    # Process each batch
    # for batch_dict in tqdm(test_dataloader, desc="Processing Validation Batches"):
    for batch_dict in test_dataloader:
        # Transfer to device if specified
        if device is not None:
            batch_dict = transfer_batch_to_device(batch_dict, device)

        # Apply time point subsampling
        if args.sample_tp is not None:
            batch_dict = subsample_obs_data(batch_dict, n_tp_to_sample=args.sample_tp)
            
        batch_size = batch_dict["obs_data"].shape[0]
        total_samples += batch_size
        
        # Compute metrics for this batch
        batch_metrics = model.compute_all_losses(
             batch_dict, n_traj, kl_coef, 
             ce_weight=ce_weight, use_joint_loss=use_joint_loss,
             z_last=z_last
        )
        
        # Collect predictions and labels for AUC calculation
        if args.classif:
            labels = batch_dict["labels"].reshape(-1)
            labels = labels.repeat(n_traj, 1)
            classif_pred = batch_metrics["label_pred"].reshape(n_traj, -1)
            all_labels.append(labels)
            all_preds.append(classif_pred)
        
        # Accumulate metrics (weighted by batch size)
        for k in cumulative_metrics:
            if k in batch_metrics and hasattr(batch_metrics[k], 'item'):
                cumulative_metrics[k] += batch_metrics[k].item() * batch_size
    
    # Calculate average metrics
    metrics = {}
    for k, v in cumulative_metrics.items():
        metrics[k] = torch.tensor(v / total_samples)
    
    # Calculate AUC using all predictions
    metrics["auc"] = 0.
    if args.classif and all_labels:
        all_labels = torch.cat(all_labels, dim=1).reshape(-1).cpu()
        all_preds = torch.cat(all_preds, dim=1).reshape(-1).cpu()

        # print("Sample predictions:", all_preds[:20])
        # print("Sample labels:", all_labels[:20])
        # print("Unique predictions:", torch.unique(all_preds))
        
        if torch.sum(all_labels) > 0 and torch.sum(all_labels) < len(all_labels):  
            # Only calculate AUC if we have both positive AND negative samples
            metrics["auc"] = sk_metrics.roc_auc_score(all_labels, all_preds)
    
    return metrics


def compute_loss_test_batches(args, model, test_dataloader, n_traj, kl_coef, 
                             ce_weight=100, use_joint_loss=True, z_last=False, device=None):
    total_samples = 0
    cumulative_metrics = {"loss": 0.0, "likelihood": 0.0, "mse": 0.0, 
                          "ce_loss": 0.0, "kl_div": 0.0, "fp_std": 0.0}
    all_labels = []
    all_preds = []
    
    # Process each batch
    for batch_dict in tqdm(test_dataloader, desc="Processing Test Set Batches"):
        if device is not None:
            batch_dict = transfer_batch_to_device(batch_dict, device)

        batch_size = batch_dict["obs_data"].shape[0]
        total_samples += batch_size
        
        # Compute metrics for this batch
        batch_metrics = model.compute_all_losses(
             batch_dict, n_traj, kl_coef, 
             ce_weight=ce_weight, use_joint_loss=use_joint_loss,
             z_last=z_last
        )
        
        # Collect predictions and labels for AUC calculation
        if args.classif:
            labels = batch_dict["labels"].reshape(-1)
            labels = labels.repeat(n_traj, 1)
            classif_pred = batch_metrics["label_pred"].reshape(n_traj, -1)
            all_labels.append(labels)
            all_preds.append(classif_pred)
        
        # Accumulate metrics (weighted by batch size)
        for k in cumulative_metrics:
            if k in batch_metrics and hasattr(batch_metrics[k], 'item'):
                cumulative_metrics[k] += batch_metrics[k].item() * batch_size
    
    # Calculate average metrics
    metrics = {}
    for k, v in cumulative_metrics.items():
        metrics[k] = torch.tensor(v / total_samples)
    
    # Calculate AUC using all predictions
    metrics["auc"] = 0.
    if args.classif and all_labels:
        all_labels = torch.cat(all_labels, dim=1).reshape(-1).cpu()
        all_preds = torch.cat(all_preds, dim=1).reshape(-1).cpu()
        
        metrics["auc"] = sk_metrics.roc_auc_score(all_labels, all_preds)
        metrics["true_y"] = all_labels
        metrics["pred_y"] = torch.sigmoid(all_preds)
    
    return metrics


def compute_loss_val_batches_distributed(args, model, test_dataloader, n_traj, kl_coef, 
                                         ce_weight=100, use_joint_loss=True, z_last=False):
    """
    Computes loss over validation data with support for distributed training
    """
    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from sklearn import metrics as sk_metrics

    total_samples = 0
    cumulative_metrics = {
        "loss": 0.0, 
        "likelihood": 0.0, 
        "mse": 0.0, 
        "ce_loss": 0.0, 
        "kl_first_p": 0.0, 
        "fp_std": 0.0
    }
    all_labels = []
    all_preds = []
    
    # Check if we're using distributed training
    is_distributed = hasattr(test_dataloader, 'sampler') and hasattr(test_dataloader.sampler, 'num_replicas')
    
    # Process each batch
    with torch.no_grad():
        for i in range(len(test_dataloader)):
            try:
                if hasattr(test_dataloader, '__next__'):
                    # For infinite generators
                    batch_dict = test_dataloader.__next__()
                else:
                    # For regular DataLoader
                    if not hasattr(test_dataloader, '_iterator') or test_dataloader._iterator is None:
                        test_dataloader._iterator = iter(test_dataloader)
                    batch_dict = next(test_dataloader._iterator)
            except (StopIteration, IndexError):
                break
            
            batch_size = batch_dict["obs_data"].shape[0]
            total_samples += batch_size
            
            # Compute metrics for this batch
            batch_metrics = model.compute_all_losses(
                 batch_dict, n_traj, kl_coef, 
                 ce_weight=ce_weight, use_joint_loss=use_joint_loss,
                 z_last=z_last
            )
            
            # Collect predictions and labels for AUC calculation
            if args.classif and "label_pred" in batch_metrics:
                labels = batch_dict["labels"].reshape(-1)
                labels = labels.repeat(n_traj, 1)
                classif_pred = batch_metrics["label_pred"].reshape(n_traj, -1)
                all_labels.append(labels)
                all_preds.append(classif_pred)
            
            # Accumulate metrics (weighted by batch size)
            for k in cumulative_metrics:
                if k in batch_metrics and hasattr(batch_metrics[k], 'item'):
                    cumulative_metrics[k] += batch_metrics[k].item() * batch_size
    
    # If running distributed, synchronize the metrics across all processes
    if is_distributed and torch.distributed.is_initialized():
        # Convert metrics to tensor
        metrics_tensor = torch.tensor([
            total_samples,
            cumulative_metrics["loss"],
            cumulative_metrics["likelihood"],
            cumulative_metrics["mse"],
            cumulative_metrics["ce_loss"],
            cumulative_metrics["kl_first_p"],
            cumulative_metrics["fp_std"]
        ], device=next(model.parameters()).device)
        
        # All-reduce to sum metrics across processes
        torch.distributed.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        # Unpack synchronized metrics
        total_samples = metrics_tensor[0].item()
        cumulative_metrics["loss"] = metrics_tensor[1].item()
        cumulative_metrics["likelihood"] = metrics_tensor[2].item()
        cumulative_metrics["mse"] = metrics_tensor[3].item()
        cumulative_metrics["ce_loss"] = metrics_tensor[4].item()
        cumulative_metrics["kl_first_p"] = metrics_tensor[5].item()
        cumulative_metrics["fp_std"] = metrics_tensor[6].item()
        
        # For AUC, gather predictions and labels from all processes
        if args.classif and all_labels:
            # This is a simplified approach - in a real implementation,
            # you would need to gather variable-sized tensors which is more complex
            pass
    
    # Calculate average metrics
    metrics = {}
    if total_samples > 0:
        for k, v in cumulative_metrics.items():
            metrics[k] = torch.tensor(v / total_samples)
    else:
        # Fallback if no samples
        for k, v in cumulative_metrics.items():
            metrics[k] = torch.tensor(0.0)
    
    # Add kl_div for compatibility
    metrics["kl_div"] = metrics["kl_first_p"] if "kl_first_p" in metrics else torch.tensor(0.0)
    
    # Calculate AUC using all predictions
    metrics["auc"] = 0.0
    if args.classif and all_labels:
        try:
            all_labels_tensor = torch.cat(all_labels, dim=1).reshape(-1).cpu()
            all_preds_tensor = torch.cat(all_preds, dim=1).reshape(-1).cpu()
            
            if torch.sum(all_labels_tensor) > 0 and torch.sum(all_labels_tensor) < len(all_labels_tensor):
                # Only calculate AUC if we have both positive and negative samples
                metrics["auc"] = sk_metrics.roc_auc_score(all_labels_tensor.numpy(), all_preds_tensor.numpy())
        except (ValueError, RuntimeError):
            # Handle potential errors in AUC calculation
            metrics["auc"] = 0.0
    
    return metrics


def set_seeds_for_distributed(base_seed, rank):
    """
    Set seeds for reproducibility in distributed training.
    Each rank gets a different seed derived from the base seed.
    
    Args:
        base_seed: The base random seed
        rank: Process rank
    """
    import torch
    import random
    import numpy as np
    
    # Each rank gets a different seed
    seed = base_seed + rank
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def log_gpu_stats(device_index=0):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_index)
        max_allocated = torch.cuda.max_memory_allocated(device_index)
        reserved = torch.cuda.memory_reserved(device_index)
        max_reserved = torch.cuda.max_memory_reserved(device_index)
        
        print(
            f"  GPU {device_index} usage - Allocated: {allocated / 1024**2:.1f} MB (Max allocated: {max_allocated / 1024**2:.1f} MB), "
            f"Reserved: {reserved / 1024**2:.1f} MB (Max Reserved: {max_reserved / 1024**2:.1f} MB)"
        )


class EarlyStopping:
    """Early stopping within a trial to avoid wasting time on plateaued models"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.mode == 'max':
            score_improved = current_score > self.best_score + self.min_delta
        else:
            score_improved = current_score < self.best_score - self.min_delta
        
        if score_improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def compute_loss_classification_only(args, model, test_dataloader, n_traj, z_last=False, device=None):
    """
    Test set evaluation for classification only. No reconstruction computation.
    """
    
    if not args.classif:
        raise ValueError("This function is only for classification models")
    
    total_samples = 0
    total_ce_loss = 0.0
    all_labels = []
    all_preds = []
    all_pids = []
    
    # Process each batch
    for batch_dict in tqdm(test_dataloader, desc="Processing Test Set (Classification Only)", file=sys.stderr):
        if device is not None:
            batch_dict = transfer_batch_to_device(batch_dict, device)

        # batch_dict = batch_postprocess(batch_dict)
        batch_size = batch_dict["obs_data"].shape[0]
        total_samples += batch_size

        if "pids" in batch_dict:
            all_pids.extend(batch_dict["pids"])
        
        # Compute classification metrics only
        batch_metrics = model.compute_classification_loss_only(
            batch_dict, n_traj=n_traj, z_last=z_last
        )
        
        # Accumulate CE loss
        total_ce_loss += batch_metrics["ce_loss"].item() * batch_size
        
        # Collect predictions and labels for AUC calculation
        labels = batch_dict["labels"].reshape(-1)
        labels = labels.repeat(n_traj, 1)
        classif_pred = batch_metrics["label_pred"].reshape(n_traj, -1)
        all_labels.append(labels)
        all_preds.append(classif_pred)
    
    # Calculate average metrics
    metrics = {}
    metrics["ce_loss"] = torch.tensor(total_ce_loss / total_samples)
    metrics["loss"] = metrics["ce_loss"]  # For classification only, loss = ce_loss
    metrics["pids"] = all_pids
    
    # Calculate AUC using all predictions
    if all_labels:
        all_labels = torch.cat(all_labels, dim=1).reshape(-1).cpu()
        all_preds = torch.cat(all_preds, dim=1).reshape(-1).cpu()
        
        metrics["auc"] = sk_metrics.roc_auc_score(all_labels, all_preds)
        metrics["true_y"] = all_labels
        metrics["pred_y"] = torch.sigmoid(all_preds)
    else:
        metrics["auc"] = 0.
    
    return metrics

