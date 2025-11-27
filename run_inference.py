import os
import argparse
import numpy as np
from loguru import logger

import torch
from torch.distributions.normal import Normal

from config import random_state, MODELS_DIR
from utils.viz_functs import parse_model_filename, format_model_filename
import utils.model_functs as utils
from utils.metrics import show_results


parser = argparse.ArgumentParser(description = 'Latent ODE Inference')

# Dataset and task parameters
parser.add_argument('--dataset', type = str, default = 'MDD', help = "dataset to load, available: MDD, CAD")
parser.add_argument('--look_back', type = str, default = '1y', help = "look back window (1y, 2y, 3y)")
parser.add_argument('--feat_type', type = str, default = 'demo_dx_med', help = "Feature type (demo_dx, demo_dx_med)")
parser.add_argument('--sample_tp', type = float, default = None, help = "number/percentage of time points to sub-sample")
parser.add_argument('--extrap', action = 'store_true', help = "use extrapolation mode (default: True)")
parser.add_argument('--batch_size', type = int, default = 64, help ="batch size")
parser.add_argument('--n_iter', type = int, default = 20, help = "number of iterations")
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
parser.add_argument('--classif_w_recon', action = 'store_true', help = "jointly consider classification loss and reconstruction loss")
parser.add_argument('--feat_dir', type = str, default = 'random_visit_w_filters_all_matched', help = "")

# Infra parameters
parser.add_argument('--cuda', type=str, default='cuda:0', help="which gpu to use")

args, unkwn = parser.parse_known_args()
args.random_seed = random_state

device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
torch.set_default_device(args.cuda)
torch.set_default_dtype(torch.float32)

# Set seed for reproducibility
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)


## Load test data loader
from data.parse_datasets import parse_datasets
from lib.create_latent_ode_model import create_LatentODE_model

logger.info("Loading data...")
data_obj = parse_datasets(args, device)
test_dataloader = data_obj["test_dataloader"]

# CREATE MODEL
logger.info("Creating model...")
obsrv_std = torch.Tensor([0.01]).to(device)
z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

# Instantiate the model with the same architecture as during training
model = create_LatentODE_model(
    args, 
    data_obj["input_dim"], 
    z0_prior, 
    obsrv_std, 
    data_obj["n_labels"],
)

## Load tuned model
# LOAD BEST MODEL STATE
model_filename = "final_latent_ode_MDD_1y_demo-dx-med_auc0p745_lr1p000e03_ntraj1_cew100_zlastTrue_gldim100_rldim100_glayer3_rlayer3_ngru100_nclassif100_sampletp0p5_20251119_0955.pt"
model_path = os.path.join(MODELS_DIR, args.dataset.lower(), model_filename)
logger.info(f"Loading model state from {model_path}")
model_state = torch.load(model_path, map_location=device)
model.load_state_dict(model_state)
model.eval()  # Set to evaluation mode


## Run inference on test set
logger.info("Running evaluation on test set...")
with torch.no_grad():
    # model_filename = os.path.basename(model_path)
    # parts = model_filename.split('_')
    parts = format_model_filename(parse_model_filename(model_path))
    
    # Extract hyperparameters from filename
    for key, value in parts.items():
       print(f"{key}: {value}")
    
    # Get test metrics
    test_metrics = utils.compute_loss_classification_only(
        args, 
        model, 
        test_dataloader,
        n_traj=parts['n_traj'],
        z_last=parts['z_last']
    )

show_results(test_metrics)

logger.info("Inference completed.")