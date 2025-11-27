#!/usr/bin/env python

# Rubanova et al.
# Latent ordinary differential equations for irregularly-sampled time series. NeurIPS, 2019
# NeurIPS, 2019

# -----

import os
import time
import copy
import tempfile
import argparse
import numpy as np
from loguru import logger
from datetime import datetime
from random import SystemRandom

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import ray
from ray import tune
from ray import train
import ray.cloudpickle as pickle
from ray.train import ScalingConfig, Checkpoint
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from config import random_state, MODELS_DIR, RESULTS_DIR
from utils.viz_functs import *
import utils.model_functs as utils
from utils.metrics import show_results
from utils.general_functs import save_pickle
from utils.data_functs import subsample_obs_data
from data.parse_datasets import parse_datasets
from lib.create_latent_ode_model import create_LatentODE_model


def evaluate_best_model_on_test_set(args, best_hyperparams, best_model_state, device):
    """
    Load the best model and evaluate it on the test set.
    """
    logger.info("="*20 + " Test Set Evaluation " + "="*20)
    
    # Update args with best hyperparameters
    for key, value in best_hyperparams.items():
        if hasattr(args, key):
            setattr(args, key, value)

    args.sample_tp = None
    
    # Load test data
    logger.info("Loading test dataset...")
    test_data_obj = parse_datasets(args, torch.device("cpu"), train_mode="test")
    test_dataloader = test_data_obj["test_dataloader"]
    
    # Create model with best hyperparameters
    logger.info("Creating model with best hyperparameters...")
    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    
    model = create_LatentODE_model(
        args, 
        test_data_obj["input_dim"], 
        z0_prior, 
        obsrv_std, 
        test_data_obj["n_labels"]
    )
    model = model.to(device)
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    with torch.no_grad():
        test_metrics = utils.compute_loss_classification_only(
            args,
            model,
            test_dataloader,
            n_traj=best_hyperparams.get("n_traj", 1),
            z_last=best_hyperparams.get("z_last", True),
            device=device
        )
    
    logger.info(f"Test Set CE Loss: {test_metrics['ce_loss']:.4f}")
    logger.info(f"Test Set AUC: {test_metrics['auc']:.4f}")
    
    # Print detailed results
    print("\n" + "="*20 + " Detailed Test Results " + "="*20)
    show_results(test_metrics)
    print("="*60 + "\n")
    
    return test_metrics


def save_test_predictions(args, test_metrics, best_hyperparams, timestamp=None):
    """
    Save test set predictions to file.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Format hyperparameters for filename
    lr_str = f"{best_hyperparams['lr']:.3e}".replace('.', 'p').replace('+','').replace('-','')
    ntraj_str = f"ntraj{best_hyperparams['n_traj']}"
    cew_str = f"cew{best_hyperparams['ce_weight']}"
    zlast_str = f"zlast{best_hyperparams['z_last']}"
    gldim_str = f"gldim{best_hyperparams['gen_latent_dims']}"
    rldim_str = f"rldim{best_hyperparams['rec_latent_dims']}"
    glayer_str = f"glayer{best_hyperparams['gen_layers']}"
    rlayer_str = f"rlayer{best_hyperparams['rec_layers']}"
    ngru_str = f"ngru{best_hyperparams['gru_units']}"
    nclassif_str = f"nclassif{best_hyperparams['classif_units']}"
    sampletp_str = f"sampletp{best_hyperparams['sample_tp']}".replace('.', 'p')
    
    model_params = f"lr{lr_str}_{ntraj_str}_{cew_str}_{gldim_str}_{rldim_str}_{glayer_str}_{rlayer_str}_{ngru_str}_{nclassif_str}_{sampletp_str}_{zlast_str}"
    
    # Create predictions filename
    feat_type_str = args.feat_type.replace('_', '-')
    test_auc_str = f"auc{test_metrics['auc']:.3f}".replace('.', 'p')
    
    pred_dir = os.path.join(RESULTS_DIR, "tuning_results", args.dataset.lower())
    os.makedirs(pred_dir, exist_ok=True)
    
    pred_file = os.path.join(
        pred_dir,
        f"test_predictions_tuned_{args.dataset.upper()}_{args.look_back}_{feat_type_str}_{test_auc_str}_{model_params}_{timestamp}.pk"
    )
    
    save_pickle(test_metrics, pred_file)
    logger.info(f"Saved test predictions to: {pred_file}")
    
    return pred_file


def train_latent_ode(config):
    """
    Ray Tune trainable function.
    """
    args = config["args"]
    hyperparams = config

    args.gen_latent_dims = hyperparams['gen_latent_dims']
    args.rec_latent_dims = hyperparams['rec_latent_dims']
    args.gen_layers = hyperparams['gen_layers']
    args.rec_layers = hyperparams['rec_layers']
    args.gru_units = hyperparams['gru_units']
    args.classif_units = hyperparams['classif_units']
    args.sample_tp = hyperparams['sample_tp']

    # DEVICE SETUP
    # Use ray.train.torch.get_device() instead of manual setup
    # This ensures the code runs on the GPU allocated by Ray Tune.
    device = train.torch.get_device()
    torch.set_default_dtype(torch.float32)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)

    # Set seeds based on Ray Train context for reproducibility across workers/trials
    worker_rank = train.get_context().get_world_rank() # Gets rank within the trial's worker group
    seed = args.random_seed + worker_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Rank {worker_rank}: Seed set to {seed}, running on device: {device}")
      
    # ============================================================================
    # Load data (Zero-Copy from Ray Object Store)
    # ============================================================================
    logger.info("Data loading from Ray Object Store")
    # Retrieve the pre-loaded data object from shared memory
    data_obj = ray.get(config["data_obj"])

    train_dataloader = data_obj["train_dataloader"] 
    val_dataloader = data_obj["val_dataloader"]
    logger.info(f"Rank {worker_rank}: Data loading finished.")

    # ============================================================================
    # Create Model, Optimizer, and Scheduler
    # ============================================================================
    logger.info("Model initiation")
    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))   # standard normal distribution
    # Instantiate the Latent ODE model
    model = create_LatentODE_model(args, data_obj["input_dim"], z0_prior, obsrv_std, data_obj["n_labels"])
    
    # Prepare Model and Optimizer with Ray Train
    model = train.torch.prepare_model(model)

    # Initialize and prepare the optimizer
    optimizer = torch.optim.Adamax(model.parameters(), lr=hyperparams["lr"])
    optimizer = train.torch.prepare_optimizer(optimizer)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Monitors validation loss
        factor=0.5,        # Factor by which LR is reduced (new_lr = lr * factor)
        patience=3,        # Number of epochs with no improvement after which LR is reduced
        min_lr=1e-5,       # Lower bound on the learning rate
        verbose=False      
    )

    # Load Checkpoint (if resuming trial)
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            checkpoint_dict = torch.load(checkpoint_path, map_location=device)

        start_epoch = checkpoint_dict["epoch"] + 1 # Start from the next epoch
        # Restore model, optimizer, and scheduler states
        # Use model.module.load_state_dict if DDP was used, but prepare_model handles this loading.
        model.load_state_dict(checkpoint_dict["model_state"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state"])
        if "scheduler_state" in checkpoint_dict:
            scheduler.load_state_dict(checkpoint_dict["scheduler_state"])
        logger.info(f"Rank {worker_rank}: Loaded checkpoint from epoch {start_epoch - 1}")


    # ============================================================================
    # Training
    # ============================================================================
    logger.info("Model training")
    logger.info(f"Rank {worker_rank}: Starting training from epoch {start_epoch} for {args.n_iter} epochs total.")
    n_epochs = args.n_iter 

    for epoch in range(start_epoch, n_epochs):
        model.train() # Set model to training mode
        total_train_loss = 0.0
        train_steps = 0

        # --- KL Coefficient Scheduling ---
        # Implements a gradual increase of the KL divergence term's weight in the loss.
        # This can help stabilize training, especially in the early stages.
        is_warmup_phase = epoch < args.warmup_epochs # Check if in classifier warmup phase
        if epoch < args.warmup_epochs:
            kl_coef = 0. # No KL penalty during classifier warm-up
        else:
            # Start increasing KL coefficient after warmup and a brief wait period
            wait_until_kl_inc = 5 # How many epochs *after warmup* to wait before starting KL increase
            kl_epoch_offset = args.warmup_epochs + wait_until_kl_inc
            if epoch < kl_epoch_offset:
                kl_coef = 0.
            else:
                # Gradually increase kl_coef towards 1
                kl_coef = (1 - 0.99 ** (epoch - kl_epoch_offset))
        # Determine whether to use the combined reconstruction + CE loss,
        # or just the classification (CE) loss during the warmup phase.
        use_joint_loss = not is_warmup_phase

        # --- Training Phase ---
        # Iterate over batches from the training dataloader
        for batch_idx, batch_dict in enumerate(train_dataloader):
            # Data is on CPU from Ray Object Store. Move to GPU here.
            batch_dict = utils.transfer_batch_to_device(batch_dict, device)
            if args.sample_tp is not None:
                batch_dict = subsample_obs_data(batch_dict, n_tp_to_sample=args.sample_tp)

            optimizer.zero_grad() # Clear previous gradients

            # Get model reference (handles DDP wrapper if present)
            # prepare_model returns the original model if num_workers=1, or the DDP-wrapped model otherwise.
            # Accessing compute_all_losses needs to be done on the underlying model instance.
            model_ref = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

            # Compute loss using hyperparameters for this trial
            train_res = model_ref.compute_all_losses(
                batch_dict,
                n_traj=hyperparams["n_traj"], # Use tuned n_traj
                kl_coef=kl_coef,
                ce_weight=hyperparams["ce_weight"], # Use tuned ce_weight
                use_joint_loss=use_joint_loss,
                z_last=hyperparams["z_last"]
            )

            loss = train_res["loss"] # Extract the main loss value
            loss.backward() # Compute gradients
            optimizer.step() # Update model parameters

            total_train_loss += loss.item() # Accumulate training loss
            train_steps += 1

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0

        # --- Validation Phase ---
        model.eval()
        val_loss = float('nan') 
        val_auc = 0.0 

        # No gradient calculations needed for validation
        with torch.no_grad():
            # Get model reference again (needed if DDP is used)
            model_ref = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model

            # Compute metrics on the validation set
            val_metrics = utils.compute_loss_val_batches(
                args, model_ref, val_dataloader,
                n_traj = hyperparams["n_traj"], 
                kl_coef = kl_coef,
                ce_weight=hyperparams["ce_weight"], 
                use_joint_loss=use_joint_loss, # Use same loss function as training for consistency
                z_last=hyperparams["z_last"],
                device=device
            )

            val_loss = val_metrics["loss"].item() 
            # Get AUC if classification is enabled and AUC was computed
            if args.classif and "auc" in val_metrics:
                val_auc = val_metrics["auc"] 

        # Update Learning Rate Scheduler based on validation loss
        if not np.isnan(val_loss):
            scheduler.step(val_loss)

        # --- Report Metrics and Checkpoint to Ray Tune ---
        current_lr = optimizer.param_groups[0]['lr'] # Get current learning rate
        metrics_to_report = {
            "loss": val_loss, # Validation loss
            "auc": val_auc, # Validation AUC
            "train_loss": avg_train_loss, # Average training loss for the epoch
            "kl_coef": kl_coef, # Current KL coefficient
            "lr": current_lr, # Current learning rate
            "epoch": epoch, # Current epoch number
        }

        # Prepare checkpoint data
        # Note: Saving raw model state dict is generally recommended when using prepare_model
        checkpoint_data = {
            "epoch": epoch,
            "model_state": model_ref.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint_path = os.path.join(temp_checkpoint_dir, 'checkpoint.pt')
            torch.save(checkpoint_data, checkpoint_path)
            
            # Create a Ray Train checkpoint from the directory
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            
            # Report metrics and checkpoint back to Ray Tune
            train.report(
                metrics_to_report, 
                checkpoint=checkpoint
            )

        # Log progress (validation set)
        if worker_rank == 0:
            logger.info(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")

    logger.info(f"Rank {worker_rank}: Finished Training")
    # del model
    # del optimizer
    # del scheduler
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Latent ODE Hyperparameter Tuning with Ray Tune')

    # Dataset and task parameters
    parser.add_argument('--dataset', type = str, default = 'CAD', help = "dataset to load, available: MDD, CAD")
    parser.add_argument('--look_back', type = str, default = '1y', help = "look back window (1y, 2y, 3y)")
    parser.add_argument('--feat_type', type = str, default = 'demo_dx', help = "Feature type (demo_dx, demo_dx_med)")
    parser.add_argument('--extrap', action = 'store_true', help = "Set extrapolation mode (default: False)")
    parser.add_argument('--batch_size', type = int, default = 64, help ="batch size")
    parser.add_argument('--sample_tp', type=float, default=None, help='Fraction of time points to sample during training')
    parser.add_argument('--n_iter', type = int, default = 100, help = "number of iterations")
    parser.add_argument('--warmup_epochs', type=int, default=0, help="Number of epochs for classifier warm-up")

    # Training and model parameters
    parser.add_argument('--gen-latent-dims', type = int, default = 100, help = "dimensionality of the latent state in the generative ODE")
    parser.add_argument('--rec-latent-dims', type = int, default = 100, help = "dimensionality of the latent state in the recognition ODE")
    parser.add_argument('--gen-layers', type = int, default = 3, help = "number of layers in ODE func in generative ODE")
    parser.add_argument('--rec-layers', type = int, default = 3, help = "number of layers in ODE func in recognition ODE")
    parser.add_argument('--units', type = int, default = 100, help = "number of units per layer in ODE func")
    parser.add_argument('--gru-units', type = int, default = 100, help = "number of units per layer in the GRU update network")
    parser.add_argument('--classif-units', type = int, default = 100, help = "number of units per layer in the classification network")
    parser.add_argument('--classif', action = 'store_false', help = "include binary classification loss")
    parser.add_argument('--classif_w_recon', action = 'store_false', help = "jointly consider classification loss and reconstruction loss")

    # Infra parameters
    parser.add_argument('--cuda', type=str, default = '2,3', help="which gpu to use if not distributed")

    # --- Ray Tune specific parameters ---
    parser.add_argument('--num_trials', type=int, default = 20, help="Number of different hyperparameter combinations (trials) to run")
    parser.add_argument('--gpus_per_trial', type=float, default = 1.0, help="Number of GPUs allocated per trial (can be fractional for time-sharing)")
    parser.add_argument('--cpus_per_trial', type=int, default = 1, help="Number of CPUs allocated per trial")
    parser.add_argument('--max_total_cpus', type=int, default = 12, help="Maximum total number of CPUs Ray should use (limits overall usage)")
    parser.add_argument('--ray_address', type=str, default = None, help="Address of Ray cluster (e.g., 'auto' or 'ray://<head_node>:10001')")

    parser.add_argument('--feat_dir', type = str, default = '', help = "")

    args, unkwn = parser.parse_known_args()
    args.random_seed = random_state

    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    logger.info(f"Setting CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}'")


    # ============================================================================
    # Initialize Ray
    # ============================================================================
    # Shutdown existing Ray instance if any, then initialize.
    if ray.is_initialized():
        ray.shutdown()
    # Connect to an existing Ray cluster or initialize locally.
    if args.ray_address:
        ray.init(address=args.ray_address, ignore_reinit_error=True)
        logger.info(f"Connected to Ray cluster at {args.ray_address}")
    else:
        # Pass num_cpus argument if provided by the user to limit total CPU usage.
        ray_init_kwargs = {"ignore_reinit_error": True}
        if args.max_total_cpus is not None:
            ray_init_kwargs["num_cpus"] = args.max_total_cpus
            logger.info(f"Initializing Ray locally, limiting total CPUs to {args.max_total_cpus}")
        else:
            logger.info("Initializing Ray locally with default CPU detection")
        ray.init(**ray_init_kwargs)

    # --- Log available resources (especially GPUs) ---
    available_resources = ray.available_resources()
    logger.info(f"Ray detected resources: {available_resources}")
    detected_gpus = available_resources.get("GPU", 0)
    detected_cpus = available_resources.get("CPU", 0)
    # Check if detected GPUs match expectations based on CUDA_VISIBLE_DEVICES
    expected_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else torch.cuda.device_count()
    if detected_gpus < expected_gpus:
        logger.warning(f"Ray detected {detected_gpus} GPUs, but CUDA_VISIBLE_DEVICES is set to '{os.environ.get('CUDA_VISIBLE_DEVICES')}'. There might be an issue with GPU detection or driver setup.")
    elif detected_gpus > 0 and args.gpus_per_trial == 1.0:
        logger.info(f"Ray will use {int(detected_gpus)} detected GPUs to run up to {int(detected_gpus)} trials in parallel.")
    elif args.gpus_per_trial > detected_gpus:
            logger.warning(f"Requested {args.gpus_per_trial} GPUs per trial, but only {detected_gpus} GPUs detected by Ray. Trials may not run.")
    if args.max_total_cpus is not None and detected_cpus > args.max_total_cpus:
        logger.warning(f"Ray detected {detected_cpus} CPUs, but limited to {args.max_total_cpus}.")
    # ---------------------------------------------------

    # ============================================================================
    # Load data
    # ============================================================================
    logger.info("Data loading")
    print("extrap:", args.extrap)
    # Load data on CPU
    data_obj = parse_datasets(args, torch.device("cpu"), train_mode="tune")
    # Put data in Ray's object store
    data_ref = ray.put(data_obj)

    # ============================================================================
    # Define Hyperparameter Search Space
    # ============================================================================
    search_space = {
        "lr": tune.loguniform(1e-3, 1e-2),
        "n_traj": tune.choice([1]), 
        "ce_weight": tune.choice([100]),
        "z_last": tune.choice([True, False]), # use last z
        "gen_latent_dims": tune.choice([100]),
        "rec_latent_dims": tune.choice([100]),
        "gen_layers": tune.choice([3]),
        "rec_layers": tune.choice([3]),
        "gru_units": tune.choice([100]),
        "classif_units": tune.choice([50, 100, 200]),
        "sample_tp": tune.choice([0.1, 0.3, 0.5, 0.7]),
        
        "args": copy.deepcopy(args),
        "data_obj": data_ref
    }

    # --- Configure Trial Scheduler ---
    # ASHA (Asynchronous Successive Halving Algorithm) does early stopping
    scheduler = ASHAScheduler(
        metric="auc",               # Metric to monitor (must match train.report key)
        mode="max",                 # Objective is to maximize validation AUC
        grace_period=10,            # Minimum number of epochs a trial runs before being stopped
        reduction_factor=4,         # 1/4 the number of trials every round
        max_t=args.n_iter           # Maximum number of epochs any trial will run
    )

    # --- Configure Resource Scaling ---
    # Defines the resources allocated to each individual trial.
    scaling_config = ScalingConfig(
        # Number of distributed workers *per trial*. 1 means single-process trial.
        # Set > 1 for DDP within each trial (requires code adjustments in train_latent_ode).
        num_workers=1,
        use_gpu=(args.gpus_per_trial > 0),
        # Specify CPU/GPU resources needed by each worker process within the trial.
        resources_per_worker={"CPU": args.cpus_per_trial, "GPU": args.gpus_per_trial}
    )

    # --- Setup TorchTrainer ---
    # This object tells Ray Tune how to execute the training function (`train_latent_ode`).
    # It bridges Ray Tune's orchestration with the PyTorch training code.
    torch_trainer = TorchTrainer(
        train_loop_per_worker=train_latent_ode,
        # train_loop_config is automatically populated by Ray Tune for each trial
        # by sampling from the `param_space` defined in `Tuner`.
        # Fixed arguments (`args`) are passed via the search_space dict.
        scaling_config=scaling_config,
    )

    # --- Setup Run Configuration ---
    # Configures aspects of the overall tuning experiment run.
    run_config = train.RunConfig(
        name=f"latent_ode_tune_{args.dataset}_auc", # Name for the experiment
        # Directory where Ray Tune will store results (logs, checkpoints)
        storage_path=os.path.join(RESULTS_DIR, "ray_results"),
        log_to_file=True, 
        # Configuration for checkpointing within trials
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=1, # Number of best checkpoints to retain per trial
            checkpoint_score_attribute="auc",
            checkpoint_score_order="max" 
        ),
        # Alternative stopping condition (e.g., stop after a certain number of iterations/epochs)
        # stop={"training_iteration": args.n_iter}
    )

    # --- Setup Tuner ---
    # The main Ray Tune API object for defining and running the HPT experiment.
    tuner = tune.Tuner(
        trainable=torch_trainer, 
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            num_samples=args.num_trials, # Total number of trials to run
            scheduler=scheduler,               # The trial scheduler (ASHA)
            # metric="auc",                      # The primary metric to optimize
            # mode="max"                         # Optimization direction (minimize loss)
        ),
        run_config=run_config # Overall run configuration
    )

    # ============================================================================
    # Run Tuning Experiment
    # ============================================================================
    logger.info(f"Starting Ray Tune with {args.num_trials} trials, optimizing for validation AUC...")
    results = tuner.fit() 

    # ============================================================================
    # Analyze Results
    # ============================================================================
    # Get the best trial result based on the specified metric and mode
    best_result = results.get_best_result(metric="auc", mode="max")

    logger.info("="*20 + " Ray Tune Results " + "="*20)
    if best_result:
        best_config = best_result.config['train_loop_config']
        # Extract the specific hyperparameters tuned
        best_hyperparams = {
            "lr": best_config.get("lr"),
            "n_traj": best_config.get("n_traj"),
            "ce_weight": best_config.get("ce_weight"),
            "z_last": best_config.get("z_last"),
            "gen_latent_dims": best_config.get("gen_latent_dims"),
            "rec_latent_dims": best_config.get("rec_latent_dims"),
            "gen_layers": best_config.get("gen_layers"),
            "rec_layers": best_config.get("rec_layers"),
            "gru_units": best_config.get("gru_units"),
            "classif_units": best_config.get("classif_units"),
            "sample_tp": best_config.get("sample_tp"),
        }
        logger.info(f"Best hyperparameters found: {best_hyperparams}")
        logger.info(f"Best trial final validation loss: {best_result.metrics['loss']:.4f}")
        logger.info(f"Best trial final validation AUC: {best_result.metrics.get('auc', 'N/A'):.4f}")
        logger.info(f"Best trial path: {best_result.path}")

        # --- Save the best model ---
        if best_result.checkpoint:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                # Construct filename based on best hyperparameters
                lr_str = f"{best_hyperparams['lr']:.3e}".replace('.', 'p').replace('+','').replace('-','')
                ntraj_str = f"ntraj{best_hyperparams['n_traj']}"
                cew_str = f"cew{best_hyperparams['ce_weight']}"
                zlast_str = f"zlast{best_hyperparams['z_last']}"
                gldim_str = f"gldim{best_hyperparams['gen_latent_dims']}"
                rldim_str = f"rldim{best_hyperparams['rec_latent_dims']}"
                glayer_str = f"glayer{best_hyperparams['gen_layers']}"
                rlayer_str = f"rlayer{best_hyperparams['rec_layers']}"
                ngru_str = f"ngru{best_hyperparams['gru_units']}"
                nclassif_str = f"nclassif{best_hyperparams['classif_units']}"
                sampletp_str = f"sampletp{best_hyperparams['sample_tp']}".replace('.', 'p')

                best_auc_score = best_result.metrics.get('auc', 0.0)
                auc_str = f"auc{best_auc_score:.3f}".replace('.', 'p')
                filename = (
                    f"best_latent_ode_{args.dataset}_{auc_str}_"
                    f"lr{lr_str}_{ntraj_str}_{cew_str}_"
                    f"{gldim_str}_{rldim_str}_{glayer_str}_"
                    f"{rlayer_str}_{ngru_str}_{nclassif_str}_"
                    f"{sampletp_str}_{zlast_str}_{timestamp}.pt"
                )
                save_path = os.path.join(MODELS_DIR, args.dataset.lower(), filename)

                # Load the state dict from the best checkpoint
                with best_result.checkpoint.as_directory() as best_checkpoint_dir:
                    checkpoint_path = os.path.join(best_checkpoint_dir, "checkpoint.pt")
                    # Load to CPU first to avoid potential GPU memory issues on the driver process
                    best_checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

                # Extract only the model state_dict
                best_model_state = best_checkpoint_dict["model_state"]

                # Save the model state dictionary
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_model_state, save_path)
                logger.info(f"Saved best model state dictionary to: {save_path}")

                # ================ TEST SET EVALUATION ================
                if args.classif and best_model_state is not None:

                    test_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    if test_device.type == 'cuda':
                        torch.cuda.empty_cache()

                    # Evaluate on test set
                    test_metrics = evaluate_best_model_on_test_set(
                        args,
                        best_hyperparams,
                        best_model_state,
                        test_device
                    )

                    # Save test predictions
                    pred_file = save_test_predictions(
                        args,
                        test_metrics,
                        best_hyperparams,
                        timestamp
                    )

                    # Log summary
                    logger.info("="*20 + " Summary " + "="*20)
                    logger.info(f"Best Validation AUC: {best_auc_score:.4f}")
                    logger.info(f"Test Set AUC: {test_metrics['auc']:.4f}")
                    logger.info(f"Model saved to: {save_path}")
                    logger.info(f"Predictions saved to: {pred_file}")

                else:
                    logger.warning("Classification is disabled or model state not available. Skipping test evaluation.")

                # ============== END TEST SET EVALUATION ==============

            except Exception as e:
                logger.error(f"Failed to save the best model or evaluate on test set: {e}")
                logger.exception(e) # Log traceback
        else:
            logger.warning("No checkpoint found for the best trial. Cannot save model or evaluate on test set.")
        # --------------------------
    else:
        logger.warning("No best trial result found. Check Ray Tune logs.")

    logger.info("="*58)


    # --- Shutdown Ray ---
    ray.shutdown()
    logger.info("Ray Tune finished.")
    print("\n--- Overall took %s mins ---" % ((time.time()-start_time)/60))
