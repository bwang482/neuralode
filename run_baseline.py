
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
from lib.create_model import create_model, print_model_summary


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

    if device.type == 'cuda':
        torch.cuda.set_device(device)
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

    print(f"\nModel type: {args.model_type}")
    print("Final hyperparameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
      
    # ============================================================================
    # Load data
    # ============================================================================
    logger.info(f"{args.dataset} cohort data loading")
    data_obj = parse_datasets(args, device, "train", rank, world_size if args.distributed else 1)
    # Returns dict data_obj: {train_dataloader, val_dataloader, test_dataloader, n_train_batches, n_val_batches, n_test_batches, n_labels}

    # ============================================================================
    # Create model
    # ============================================================================
    logger.info(f"Model initiation: {args.model_type}")
    obsrv_std = torch.Tensor([args.obsrv_std]).to(device) 
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))   # standard normal distribution

    # Create model using factory function (supports all model types)
    model = create_model(args, data_obj["input_dim"], z0_prior, obsrv_std, data_obj["n_labels"])
    model = model.to(device)

    # Print model summary (only on main process)
    if rank == 0 or not args.distributed:
        print_model_summary(model, args.model_type)

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

    optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)   # adamax algorithm
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Assuming you minimize a metric like validation loss
        factor=0.5,        # Reduce LR by factor of 0.5
        patience=5,        
        min_lr=1e-5,
        verbose=True       # Print updates whenever LR is reduced
    )
    
    # ============================================================================
    # Early stopping setup
    # ============================================================================
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(MODELS_DIR, args.dataset.lower(), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Model training")
    logger.info(f"Early stopping patience: {args.early_stop_patience} epochs")
    logger.info(f"Validation sample_tp: {args.val_sample_tp}")
    model.train()

    print("Initial learning rate:", args.lr)
    
    val_args = copy.deepcopy(args) # Create validation args
    val_args.sample_tp = args.val_sample_tp # Use val_sample_tp for validation
    
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
                        val_args, model.module if isinstance(model, DDP) else model, 
                        data_obj["val_dataloader"],
                        n_traj=args.n_traj, 
                        kl_coef=kl_coef,
                        ce_weight=args.ce_weight,
                        use_joint_loss=use_joint_loss,
                        z_last=args.z_last
                    )   # evaluation on test data
                else:
                    val_res = utils.compute_loss_val_batches(
                        val_args, model, data_obj["val_dataloader"],
                        n_traj = args.n_traj, kl_coef = kl_coef,
                        ce_weight=args.ce_weight,
                        use_joint_loss=use_joint_loss,
                        z_last=args.z_last,
                        device=device
                    )   # evaluation on test data

                scheduler.step(val_res["loss"].detach())

                current_val_auc = float(val_res["auc"])

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

                # ============================================================================
                # Early stopping check
                # ============================================================================
                if current_val_auc > best_val_auc:
                    best_val_auc = current_val_auc
                    best_epoch = current_epoch
                    epochs_without_improvement = 0
                    
                    # Save best model state
                    best_model_state = copy.deepcopy(
                        model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    )
                    
                    if should_print:
                        print(f"*** New best validation AUC: {best_val_auc:.4f} at epoch {best_epoch} ***")
                        
                        # Save checkpoint
                        if args.save_file:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                            model_type_str = args.model_type.replace('_', '-')
                            checkpoint_path = os.path.join(
                                checkpoint_dir,
                                f"best_checkpoint_{model_type_str}_{args.dataset}_{args.look_back}_epoch{best_epoch}_auc{best_val_auc:.4f}_{timestamp}.pt"
                            )
                            torch.save({
                                'epoch': best_epoch,
                                'model_state_dict': best_model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_auc': best_val_auc,
                                'model_type': args.model_type,
                                'args': args
                            }, checkpoint_path)
                            print(f"Checkpoint saved to: {checkpoint_path}")
                else:
                    epochs_without_improvement += 1
                    if should_print:
                        print(f"No improvement for {epochs_without_improvement} epoch(s). Best AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                
                # Check if we should stop
                if epochs_without_improvement >= args.early_stop_patience:
                    if should_print:
                        print(f"\n*** Early stopping triggered after {current_epoch + 1} epochs ***")
                        print(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}")
                    break

        model.train()  # switch back to training mode

    # # Create the memory snapshot file
    # export_memory_snapshot()
    # # Stop recording memory snapshot history
    # stop_record_memory_history()

    print("--- Training took %s mins ---\n" % ((time.time()-start_time)/60))

    # ============================================================================
    # Restore best model
    # ============================================================================
    if best_model_state is not None:
        if isinstance(model, DDP):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {best_epoch} with validation AUC {best_val_auc:.4f}")


    # ============================================================================
    # Show validation set results (using best model)
    # ============================================================================
    logger.info("="*20 + " Validation set results " + "="*20)
    final_hyperparams = {
        "model_type": args.model_type,
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
        "val_sample_tp": args.val_sample_tp,
        "obsrv_std": args.obsrv_std,
        "early_stop_patience": args.early_stop_patience,
        "best_epoch": best_epoch,
    }
    
    # Add RNN-VAE specific hyperparameters if applicable
    if args.model_type == 'rnn_vae':
        final_hyperparams.update({
            "decoder_type": args.decoder_type,
            "bidirectional": args.bidirectional,
            "use_attention": args.use_attention,
            "dec_layers": args.dec_layers,
        })
    
    val_loss = float(val_res["loss"].detach())
    val_auc = best_val_auc  # Use best validation AUC
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Hyperparameters: {final_hyperparams}")
    logger.info(f"Validation set final loss: {val_loss:.4f}")
    logger.info(f"Validation set best AUC: {val_auc:.4f} (at epoch {best_epoch})")
    
    # Save model
    if args.save_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_type_str = args.model_type.replace('_', '-')
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
        
        # Include model type in filename
        filename = f"final_{model_type_str}_{args.dataset}_{args.look_back}_{featype_str}_{auc_str}_lr{lr_str}_" + \
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
            f"test_predictions_{model_type_str}_{args.look_back}_{featype_str}_{test_auc_str}_{model_params}_{timestamp}.pk"
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
    
    # Validate model type
    valid_model_types = ['latent_ode', 'latent_ode_rnn', 'rnn_vae', 'ode_rnn', 'classic_rnn']
    if args.model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type: {args.model_type}. Choose from: {valid_model_types}")
    
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent ODE / RNN-VAE Training')

    # Model selection
    parser.add_argument('--model_type', type=str, default='latent_ode',
                        choices=['latent_ode', 'latent_ode_rnn', 'rnn_vae', 'ode_rnn', 'classic_rnn'],
                        help="Model type: 'latent_ode' (VAE, ODE-RNN enc + ODE dec), "
                             "'latent_ode_rnn' (VAE, RNN enc + ODE dec), "
                             "'rnn_vae' (VAE, RNN enc + RNN dec), "
                             "'ode_rnn' (direct, ODE-RNN enc), "
                             "'classic_rnn' (direct, RNN enc)")

    # Dataset and task parameters
    parser.add_argument('--dataset', type=str, default='CAD', help="dataset to load, available: MDD, CAD")
    parser.add_argument('--look_back', type=str, default='1y', help="look back window (1y, 2y, 3y)")
    parser.add_argument('--feat_type', type=str, default='demo_dx', help="Feature type (demo_dx, demo_dx_med)")
    parser.add_argument('--sample_tp', type=float, default=None, help="number/percentage of time points to sub-sample")
    parser.add_argument('--val_sample_tp', type=float, default=None, help="sample_tp for validation set")
    parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode (default: False)")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--n_iter', type=int, default=100, help="number of iterations")
    parser.add_argument('--lr', type=float, default=1e-3, help="starting learning rate")
    parser.add_argument('--n_traj', type=int, default=1, help="number of latent trajectories")
    parser.add_argument('--ce_weight', type=int, default=100, help="weight multiplied to cross-entropy loss")
    parser.add_argument('--z_last', action='store_false', help="use the latent state of the last time point for prediction (default: True)")
    parser.add_argument('--obsrv_std', type=float, default=0.01, help="observation noise std for reconstruction loss")
    parser.add_argument('--early_stop_patience', type=int, default=10, help="number of epochs without improvement before stopping")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay (L2 penalty)")

    # Model architecture parameters
    parser.add_argument('--gen-latent-dims', type=int, default=100, help="dimensionality of the latent state in the generative ODE")
    parser.add_argument('--rec-latent-dims', type=int, default=100, help="dimensionality of the latent state in the recognition ODE")
    parser.add_argument('--gen-layers', type=int, default=3, help="number of layers in ODE func in generative ODE")
    parser.add_argument('--rec-layers', type=int, default=3, help="number of layers in ODE func in recognition ODE")
    parser.add_argument('--units', type=int, default=100, help="number of units per layer in ODE func")
    parser.add_argument('--gru-units', type=int, default=100, help="number of units per layer in the GRU update network")
    parser.add_argument('--classif-units', type=int, default=100, help="number of units per layer in the classification network")
    parser.add_argument('--dropout-rate', type=float, default=0.0, help="dropout rate in the classifier network")
    parser.add_argument('--use_traj_attention', action='store_true', help='Use attention over trajectory for classification')

    # RNN-VAE specific parameters (NEW)
    parser.add_argument('--decoder_type', type=str, default='simple',
                        choices=['simple', 'standard', 'input_driven'],
                        help="RNN decoder type for RNN-VAE model")
    parser.add_argument('--bidirectional', action='store_true', 
                        help="Use bidirectional encoder (for latent_ode_rnn, rnn_vae, classic_rnn)")
    parser.add_argument('--use_attention', action='store_true', 
                        help="Use attention mechanism in encoder (for latent_ode_rnn and rnn_vae)")
    parser.add_argument('--dec_layers', type=int, default=1, 
                        help="Number of decoder layers (for rnn_vae)")
    parser.add_argument('--rnn_layers', type=int, default=1,
                        help="Number of RNN layers (for classic_rnn)")

    # Classification parameters
    parser.add_argument('--classif', action='store_false', help="include binary classification loss")
    parser.add_argument('--classif_w_recon', action='store_false', help="jointly consider classification loss and reconstruction loss")
    parser.add_argument('--warmup_epochs', type=int, default=0, help="Number of epochs for classifier warm-up")

    # Distributed training and save file parameters
    parser.add_argument('--distributed', action='store_true', help="enable distributed training")
    parser.add_argument('--world_size', type=int, default=2, help="number of processes (GPUs) for distributed training")
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1], help="specific GPU device IDs to use (e.g., 0 1 for cuda:0 and cuda:1)")
    parser.add_argument('--cuda', type=str, default='cuda:0', help="which gpu to use if not distributed")
    parser.add_argument('--save_file', action='store_false', help="save trained model")

    parser.add_argument('--feat_dir', type=str, default='', help="")

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
        logger.info(f"Model type: {args.model_type}")
        # Single process training
        main(0, 1, args) 
