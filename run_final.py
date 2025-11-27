#!/usr/bin/env python

# Rubanova et al.
# Latent ordinary differential equations for irregularly-sampled time series. NeurIPS, 2019
# NeurIPS, 2019

# -----

import os
import time
import copy
import argparse
import numpy as np
from loguru import logger
from datetime import datetime
from random import SystemRandom

import torch
from torch.distributions.normal import Normal
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from config import MODELS_DIR, RESULTS_DIR, random_state
from utils.viz_functs import *
import utils.model_functs as utils
from utils.metrics import show_results
from utils.general_functs import dt_setup, dt_cleanup, save_pickle
from data.parse_datasets import parse_datasets
from lib.create_latent_ode_model import create_LatentODE_model


def main(rank, world_size, args):
    # Setup distributed process
    if args.distributed:
        dt_setup(rank, world_size)
        # Use the specific GPUs from the device_ids list based on rank
        device_id = args.device_ids[rank]
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
        torch.set_default_device(device)
    else:
        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        # torch.set_default_device(args.cuda)

    # torch.cuda.set_device(device)
    torch.set_default_dtype(torch.float32)

    # Set seeds for deterministic behavior
    torch.manual_seed(args.random_seed + rank if args.distributed else args.random_seed) # set the seed for generating random numbers
    np.random.seed(args.random_seed + rank if args.distributed else args.random_seed) # set numpy random seed

    # Print process info
    if args.distributed:
        print(f"Running on rank {rank}/{world_size} using {device}")
    else:
        print(f"Running on {device} in non-distributed mode")

    # Start recording memory snapshot history
    # start_record_memory_history()

    print("Final hyperparameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
      
    # ============================================================================
    # Load data
    # ============================================================================
    logger.info(f"{args.dataset} cohort data loading")
    data_obj = parse_datasets(args, device, "train", rank, world_size if args.distributed else 1)
    # Returns dict data_obj: {train_dataloader, val_dataloader, test_dataloader, n_train_batches, n_val_batches, n_test_batches, n_labels}

    # ============================================================================
    # Create model
    # ============================================================================
    logger.info("Model initiation")
    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))   # standard normal distribution

    model = create_LatentODE_model(args, data_obj["input_dim"], z0_prior, obsrv_std, data_obj["n_labels"])   # create latent ODE model
    model = model.to(device)

    # Wrap model in DDP if using distributed training
    if args.distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=True)
        # Synchronize all processes
        dist.barrier()


    # ============================================================================
    # Training
    # ============================================================================
    start_time = time.time()
    n_batch = data_obj["n_train_batches"]

    optimizer = torch.optim.Adamax(model.parameters(), lr = args.lr)   # adamax algorithm (a variant of adam based on infinity norm); lr = learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Assuming you minimize a metric like validation loss
        factor=0.5,        # Reduce LR by factor of 0.5
        patience=5,        
        min_lr=1e-5,
        verbose=True       # Print updates whenever LR is reduced
    )

    logger.info("Model training")
    model.train()

    print("Initial learning rate:", args.lr)
    for itr in range(1, n_batch * (args.n_iter + 1)):
        current_epoch = (itr - 1) // n_batch
        is_warmup_phase = current_epoch < args.warmup_epochs

        if args.distributed and (itr - 1) % n_batch == 0:
            # Set epoch for DistributedSampler at the beginning of each epoch
            data_obj["train_dataloader"].sampler.set_epoch(current_epoch)

        # utils.log_gpu_stats(device) # logging GPU usage

        optimizer.zero_grad()   # reset the gradients of all optimized tensors to None

        # --- KL Coefficient Scheduling ---
        if current_epoch < args.warmup_epochs:
            kl_coef = 0. # No KL penalty during classifier warm-up
        else:
            wait_until_kl_inc = 5 # How many epochs *after warmup* to wait
            kl_epoch_offset = args.warmup_epochs + wait_until_kl_inc
            if current_epoch < kl_epoch_offset:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (current_epoch - kl_epoch_offset))

        use_joint_loss = not is_warmup_phase # Use joint loss or CE loss only

        if args.distributed: # Multi-GPU distributed mode
            batch_dict = utils.get_next_batch_distributed(data_obj["train_dataloader"])   # next batch of samples

            # Ensure model.module is used in distributed mode for compute_all_losses
            if isinstance(model, DDP):
                train_res = model.module.compute_all_losses(
                    batch_dict, n_traj=args.n_traj, kl_coef=kl_coef, 
                    ce_weight=args.ce_weight, use_joint_loss=use_joint_loss, z_last=args.z_last
                )
            else:
                train_res = model.compute_all_losses(
                    batch_dict, n_traj=args.n_traj, kl_coef=kl_coef, 
                    ce_weight=args.ce_weight, use_joint_loss=use_joint_loss, z_last=args.z_last
                )
        else: # Single-GPU mode
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"])   # next batch of samples
            train_res = model.compute_all_losses(
                batch_dict, 
                n_traj=args.n_traj, 
                kl_coef=kl_coef,
                ce_weight=args.ce_weight,
                use_joint_loss=use_joint_loss, # joint loss or only CE loss
                z_last=args.z_last
            )

        train_res["loss"].backward()   # compute the gradient of current tensor wrt graph leaves w.r.t. loss
        optimizer.step()   # performs a single optimization step (parameter update)

        # Clear cache periodically to prevent memory buildup
        # if itr % 10 == 0:
        # 	print("Clearing cache...")
        # 	torch.cuda.empty_cache()

        n_iter_to_eval = 1
        if itr % (n_iter_to_eval * n_batch) == 0:
            # Only log in the main process if distributed
            should_print = not args.distributed or (args.distributed and rank == 0)

            if args.distributed:
                # Synchronize before evaluation
                dist.barrier()

            # --- Validation set evaluation ---
            model.eval() # Switch to evaluation mode
            with torch.no_grad():

                if args.distributed:
                    val_res = utils.compute_loss_val_batches_distributed(
                        args, model.module if isinstance(model, DDP) else model, 
                        data_obj["val_dataloader"],
                        n_traj=args.n_traj, 
                        kl_coef=kl_coef,
                        ce_weight=args.ce_weight,
                        use_joint_loss=use_joint_loss,
                        z_last=args.z_last
                    )   # evaluation on test data
                else:
                    val_res = utils.compute_loss_val_batches(
                        args, model, data_obj["val_dataloader"],
                        n_traj = args.n_traj, kl_coef = kl_coef,
                        ce_weight=args.ce_weight,
                        use_joint_loss=use_joint_loss,
                        z_last=args.z_last,
                        device=device
                    )   # evaluation on test data

                scheduler.step(val_res["loss"].detach())
                if should_print:
                    message = 'Epoch {:04d} | Loss {:.3f} | LL {:.3f} | CE Loss {:.3f} | MSE {:.3f} | AUC {:.3f} | KL {:.3f} | FP STD {:.3f} |'.format(
                        current_epoch, val_res["loss"].detach(), val_res["likelihood"].detach(), val_res["ce_loss"].detach(),
                            val_res["mse"].detach(), val_res["auc"], val_res["kl_div"], val_res["fp_std"])
                
                    print(message)

                    # Print the current learning rate
                    print("kl_coef:", kl_coef)
                    # current_lr = optimizer.param_groups[0]['lr']
                    # print(f'Epoch [{current_epoch}] - Learning Rate: {current_lr:.6f}')
                    # print("- So far took %s mins -" % ((time.time()-start_time)/60))

        model.train()  # switch back to training mode

    # # Create the memory snapshot file
    # export_memory_snapshot()
    # # Stop recording memory snapshot history
    # stop_record_memory_history()

    print("--- Training took %s mins ---\n" % ((time.time()-start_time)/60))

    # ============================================================================
    # Show validation set results
    # ============================================================================
    logger.info("="*20 + " Validation set results " + "="*20)
    final_hyperparams = {
        "lr": args.lr,
        "n_traj": args.n_traj,
        "ce_weight": args.ce_weight,
        "z_last": args.z_last,
        "gen_latent_dims": args.gen_latent_dims,
        "rec_latent_dims": args.rec_latent_dims,
        "gen_layers": args.gen_layers,
        "rec_layers": args.rec_layers,
        "gru_units": args.gru_units,
        "classif_units": args.classif_units,
        "sample_tp": args.sample_tp,
    }
    val_loss = float(val_res["loss"].detach())
    val_auc = float(val_res["auc"])
    logger.info(f"Hyperparameters: {final_hyperparams}")
    logger.info(f"Validation set final loss: {val_loss:.4f}")
    logger.info(f"Validation set final AUC: {val_auc:.4f}")
    # Save model
    if args.save_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        lr_str = f"{final_hyperparams['lr']:.3e}".replace('.', 'p').replace('+','').replace('-','') 
        ntraj_str = f"ntraj{final_hyperparams['n_traj']}"
        cew_str = f"cew{final_hyperparams['ce_weight']}"
        zlast_str = f"zlast{final_hyperparams['z_last']}"
        gldim_str = f"gldim{final_hyperparams['gen_latent_dims']}"
        rldim_str = f"rldim{final_hyperparams['rec_latent_dims']}"
        glayer_str = f"glayer{final_hyperparams['gen_layers']}"
        rlayer_str = f"rlayer{final_hyperparams['rec_layers']}"
        ngru_str = f"ngru{final_hyperparams['gru_units']}"
        nclassif_str = f"nclassif{final_hyperparams['classif_units']}"
        sampletp_str = f"sampletp{final_hyperparams['sample_tp']}".replace('.', 'p')

        featype_str = args.feat_type.replace('_', '-')
        auc_str = f"auc{val_auc:.3f}".replace('.', 'p')
        filename = f"final_latent_ode_{args.dataset}_{args.look_back}_{featype_str}_{auc_str}_lr{lr_str}_" + \
                    f"{ntraj_str}_{cew_str}_{gldim_str}_{rldim_str}_{glayer_str}_{rlayer_str}_{ngru_str}_{nclassif_str}_{sampletp_str}_{zlast_str}_{timestamp}.pt"
        save_path = os.path.join(MODELS_DIR, args.dataset.lower(), filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(model.module.state_dict() if isinstance(model, DDP) else model.state_dict(), save_path)
        logger.info(f"Final model saved to: {save_path}")
    logger.info("="*58)

    # ============================================================================
    # Test set inference and results
    # ============================================================================
    print("\n")
    logger.info("="*20 + " Final test set results " + "="*20)
    model.eval()
    with torch.no_grad():

        test_args = copy.deepcopy(args)
        test_args.sample_tp = None  # sample_tp is None during final test set evaluation
        
        # Load test data
        test_data = parse_datasets(test_args, device, "test", 0, 1)

        # Compute test metrics
        args.sample_tp = None  # sample_tp is None during final test set evaluation
        test_metrics = utils.compute_loss_classification_only(
            test_args, 
            model.module if isinstance(model, DDP) else model,
            test_data["test_dataloader"],
            n_traj=test_args.n_traj,
            z_last=test_args.z_last,
            device=device
        )

    # Print test results
    show_results(test_metrics)

    # Save predictions
    if args.save_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_params = f"lr{lr_str}_{ntraj_str}_{cew_str}_{gldim_str}_{rldim_str}_{glayer_str}_{rlayer_str}_" + \
                            f"{ngru_str}_{nclassif_str}_{sampletp_str}_{zlast_str}"
        test_auc_str = f"auc{test_metrics['auc']:.3f}".replace('.', 'p')
        pred_dir = os.path.join(RESULTS_DIR, "final_results", args.dataset.upper())

        # Create directory if it doesn't exist
        os.makedirs(pred_dir, exist_ok=True)

        pred_file = os.path.join(
            pred_dir,
            f"test_predictions_{args.look_back}_{featype_str}_{test_auc_str}_{model_params}_{timestamp}.pk"
        )

        save_pickle(test_metrics, pred_file)
        logger.info(f"\nSaved test predictions to: {pred_file}")

    logger.info("="*58)
    # ============== END OF TEST SET EVALUATION ==============

    # Clean up distributed processes
    if args.distributed:
        dt_cleanup()

    print("\n--- Overall took %s mins ---" % ((time.time()-start_time)/60))


def validate_args(args):
    """Validate command line arguments"""
    if args.distributed:
        # Check if device_ids has the correct length
        if len(args.device_ids) != args.world_size:
            raise ValueError(f"Number of device IDs ({len(args.device_ids)}) must match world_size ({args.world_size})")
        
        # Make sure all specified device IDs are available
        available_devices = torch.cuda.device_count()
        for device_id in args.device_ids:
            if device_id >= available_devices:
                raise ValueError(f"Device cuda:{device_id} is not available. Only {available_devices} GPUs detected.")
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Latent ODE')   # parser for command-line options and arguments

    # Dataset and task parameters
    parser.add_argument('--dataset', type = str, default = 'CAD', help = "dataset to load, available: MDD, CAD")
    parser.add_argument('--look_back', type = str, default = '1y', help = "look back window (1y, 2y, 3y)")
    parser.add_argument('--feat_type', type = str, default = 'demo_dx', help = "Feature type (demo_dx, demo_dx_med)")
    parser.add_argument('--sample_tp', type = float, default = None, help = "number/percentage of time points to sub-sample")
    parser.add_argument('--extrap', action = 'store_true', help = "Set extrapolation mode (default: False)")

    # Training parameters
    parser.add_argument('--batch_size', type = int, default = 64, help ="batch size")
    parser.add_argument('--n_iter', type = int, default = 100, help = "number of iterations")
    parser.add_argument('--lr', type = float, default = 1e-3, help = "starting learning rate")
    parser.add_argument('--n_traj', type = int, default = 1, help = "number of latent trajectories")
    parser.add_argument('--ce_weight', type = int, default = 100, help = "weight multiplied to cross-entropy loss")
    parser.add_argument('--z_last', action = 'store_false', help = "use the latent state of the last time point for prediction (default: True)")

    # Model parameters
    parser.add_argument('--gen-latent-dims', type = int, default = 100, help = "dimensionality of the latent state in the generative ODE")
    parser.add_argument('--rec-latent-dims', type = int, default = 100, help = "dimensionality of the latent state in the recognition ODE")
    parser.add_argument('--gen-layers', type = int, default = 3, help = "number of layers in ODE func in generative ODE")
    parser.add_argument('--rec-layers', type = int, default = 3, help = "number of layers in ODE func in recognition ODE")
    parser.add_argument('--units', type = int, default = 100, help = "number of units per layer in ODE func")
    parser.add_argument('--gru-units', type = int, default = 100, help = "number of units per layer in the GRU update network")
    parser.add_argument('--classif-units', type = int, default = 100, help = "number of units per layer in the classification network")

    # Classification parameters
    parser.add_argument('--classif', action = 'store_false', help = "include binary classification loss")
    parser.add_argument('--classif_w_recon', action = 'store_false', help = "jointly consider classification loss and reconstruction loss")
    parser.add_argument('--warmup_epochs', type = int, default = 0, help = "Number of epochs for classifier warm-up")

    # Distributed training and save file parameters
    parser.add_argument('--distributed', action = 'store_true', help = "enable distributed training")
    parser.add_argument('--world_size', type = int, default = 2, help = "number of processes (GPUs) for distributed training")
    parser.add_argument('--device_ids', nargs = '+', type = int, default = [0, 1], help = "specific GPU device IDs to use (e.g., 0 1 for cuda:0 and cuda:1)")
    parser.add_argument('--cuda', type = str, default = 'cuda:0', help = "which gpu to use if not distributed")
    parser.add_argument('--save_file', action = 'store_false', help = "save trained model")

    parser.add_argument('--feat_dir', type = str, default = '', help = "")

    args, unkwn = parser.parse_known_args()
    args.random_seed = random_state

    # Validate arguments
    args = validate_args(args)

    if args.distributed:
        # Configure env vars for specific GPUs
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.device_ids))
        logger.info(f"Using GPUs: {args.device_ids}")

        # Launch multiple processes for distributed training
        world_size = args.world_size
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        logger.info(f"Using GPU: {args.cuda}")
        # Single process training
        main(0, 1, args) 

