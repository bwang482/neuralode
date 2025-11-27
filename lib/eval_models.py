# -----

import os
import sys
import time
import argparse
from random import SystemRandom

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import matplotlib
import matplotlib.pyplot as plt

# -----

sys.path.append("/data/tge/Tian/NeuralODE/scripts/LatentODE/toy")
sys.path.append("/data/tge/Tian/NeuralODE/scripts/LatentODE/github")

# -----

import utils_toy
from generate_toy import generate_toy


parser = argparse.ArgumentParser(description = 'Toy Model')   # parser for command-line options and arguments

parser.add_argument('--n', type = int, default = 1000, help = "number of samples")
parser.add_argument('--n_tp', type = int, default = 100, help = "total number of time points")
parser.add_argument('--max-t', type = float, default = 5., help = "maximum time interval")
parser.add_argument('--extrap', action = 'store_false', help = "set extrapolation mode; default = extrapolation mode")
parser.add_argument('--sample-tp', type = float, default = 25, help = "number/percentage of time points to sub-sample")
parser.add_argument('--noise-weight', type = float, default = 0.1, help = "noise amplitude added to trajectories")
parser.add_argument('--batch-size', type = int, default = 50, help = "batch size")
parser.add_argument('--random-seed', type = int, default = 2024, help = "random seed")

args_toy, unkwn = parser.parse_known_args()

# -----

import utils
from create_latent_ode_model import create_LatentODE_model


parser = argparse.ArgumentParser(description = 'Latent ODE')   # parser for command-line options and arguments

parser.add_argument('--n_iter', type = int, default = 100, help = "number of iterations")
parser.add_argument('--lr', type = float, default = 1e-2, help = "starting learning rate")
parser.add_argument('--random-seed', type = int, default = 2024, help = "random seed")

parser.add_argument('--gen-latent-dims', type = int, default = 10, help = "dimensionality of the latent state in the generative ODE")
parser.add_argument('--rec-latent-dims', type = int, default = 20, help = "dimensionality of the latent state in the recognition ODE")
parser.add_argument('--gen-layers', type = int, default = 1, help = "number of layers in ODE func in generative ODE")
parser.add_argument('--rec-layers', type = int, default = 1, help = "number of layers in ODE func in recognition ODE")
parser.add_argument('--units', type = int, default = 100, help = "number of units per layer in ODE func")
parser.add_argument('--gru-units', type = int, default = 100, help = "number of units per layer in the GRU update network")
parser.add_argument('--classif-units', type = int, default = 100, help = "number of units per layer in the classification network")

parser.add_argument('--classif', action = 'store_false', help = "include binary classification loss")
parser.add_argument('--classif_w_recon', action = 'store_false', help = "jointly consider classification loss and reconstruction loss")

parser.add_argument('--save-file', type = str, default = '/data/tge/Tian/NeuralODE/scripts/LatentODE/slurm/ode', help = "directory abd file name of the model parameters")

args, unkwn = parser.parse_known_args()

# -----

if torch.cuda.is_available():
        device = 'cuda'
else:
        device = 'cpu'

torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

print("--- %s ---" % device)

# -----

torch.manual_seed(args_toy.random_seed)   # set the seed for generating random numbers
np.random.seed(args_toy.random_seed)   # set numpy random seed

data_obj = generate_toy(args_toy, device)   # generate toy datasets

# -----

torch.manual_seed(args.random_seed)   # set the seed for generating random numbers
np.random.seed(args.random_seed)   # set numpy random seed

obsrv_std = torch.tensor([0.01])

z0_prior = Normal(torch.tensor([0.]), torch.tensor([1.]))   # standard normal distribution

model = create_LatentODE_model(args, data_obj["input_dim"], z0_prior, obsrv_std, data_obj["n_label"])   # create latent ODE model

model.load_state_dict(torch.load('/data/tge/Tian/NeuralODE/scripts/LatentODE/slurm/ode', weights_only=True))

# -----

kl_coef = 1 - 0.99 ** (args.n_iter - 10)

test_res = utils.compute_loss_all_batches(args, model, data_obj["test_dataloader"], n_traj = 3, kl_coef = kl_coef)   # evaluation on the test data

message = 'Epoch {:04d} | Loss {:.3f} | LL {:.3f} | CE Loss {:.3f} | MSE {:.3f} | AUC {:.3f} | KL {:.3f} | FP STD {:.3f} |'.format(
		args.n_iter, test_res["loss"].detach(), test_res["likelihood"].detach(), test_res["ce_loss"].detach(),
		test_res["mse"].detach(), test_res["auc"], test_res["kl_div"], test_res["fp_std"])

print(message)



