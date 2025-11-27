#!/usr/bin/env python

# Rubanova et al.
# Latent ordinary differential equations for irregularly-sampled time series. NeurIPS, 2019
# NeurIPS, 2019

# -----

import os
import sys
import time
import argparse
from loguru import logger
from random import SystemRandom

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
# from torch.autograd.profiler import record_function

from config import random_state
from utils.viz_functs import *
import utils.model_functs as utils
from data.parse_datasets import parse_datasets
# from data.parse_datasets_v0 import parse_datasets
from lib.create_latent_ode_model import create_LatentODE_model



parser = argparse.ArgumentParser(description = 'Latent ODE')   # parser for command-line options and arguments

parser.add_argument('--dataset', type = str, default = 'rpdrml', help = "Dataset to load. Available: rpdrml")
parser.add_argument('--sample_tp', type = float, default = None, help = "number/percentage of time points to sub-sample")
parser.add_argument('--extrap', action = 'store_false', help = "set extrapolation mode; default = extrapolation mode")

parser.add_argument('--batch_size', type = int, default = 64, help ="batch size")
parser.add_argument('--n_iter', type = int, default = 10, help = "number of iterations")
parser.add_argument('--lr', type = float, default = 1e-2, help = "starting learning rate")
parser.add_argument('--n_traj', type = int, default = 3, help = "number of latent trajectories")

parser.add_argument('--gen-latent-dims', type = int, default = 10, help = "dimensionality of the latent state in the generative ODE")
parser.add_argument('--rec-latent-dims', type = int, default = 20, help = "dimensionality of the latent state in the recognition ODE")
parser.add_argument('--gen-layers', type = int, default = 1, help = "number of layers in ODE func in generative ODE")
parser.add_argument('--rec-layers', type = int, default = 1, help = "number of layers in ODE func in recognition ODE")
parser.add_argument('--units', type = int, default = 100, help = "number of units per layer in ODE func")
parser.add_argument('--gru-units', type = int, default = 100, help = "number of units per layer in the GRU update network")
parser.add_argument('--classif-units', type = int, default = 100, help = "number of units per layer in the classification network")

parser.add_argument('--classif', action = 'store_false', help = "include binary classification loss")
parser.add_argument('--classif_w_recon', action = 'store_false', help = "jointly consider classification loss and reconstruction loss")

parser.add_argument('--cuda', type = str, default = 'cuda:3', help = "which gpu to use")
parser.add_argument('--save-file', type = str, default = '/data/tge/Tian/NeuralODE/scripts/LatentODE/slurm/ode', help = "directory abd file name of the model parameters")

args, unkwn = parser.parse_known_args()


device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
torch.set_default_device(args.cuda)
torch.set_default_dtype(torch.float32)


if __name__ == '__main__':
	# Start recording memory snapshot history
	# start_record_memory_history()
      
	### LOAD DATA
	args.random_seed = random_state
	torch.manual_seed(random_state)   # set the seed for generating random numbers
	np.random.seed(random_state)   # set numpy random seed

	logger.info("Data loading")
	data_obj = parse_datasets(args, device)
	# return dict data_obj: {train_dataloader, test_dataloader, n_train_batches, n_test_batches, input_dim, n_labels}


	### CREATE MODEL
	logger.info("Model initiation")
	obsrv_std = torch.Tensor([0.01]).to(device)
	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))   # standard normal distribution

	model = create_LatentODE_model(args, data_obj["input_dim"], z0_prior, obsrv_std, data_obj["n_labels"])   # create latent ODE model


	### TRAINING
	start_time = time.time()

	optimizer = torch.optim.Adamax(model.parameters(), lr = args.lr)   # adamax algorithm (a variant of adam based on infinity norm); lr = learning rate
	n_batch = data_obj["n_train_batches"]

	logger.info("Model training")
	model.train() 
	for itr in range(1, n_batch * (args.n_iter + 1)):   # each pass of the dataset takes n_batch updates
		utils.log_gpu_stats(device)

		optimizer.zero_grad()   # reset the gradients of all optimized tensors to None
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)   # decay the current learning rate by a factor of 0.999

		wait_until_kl_inc = 10   # num of iterations until KL divergence is added to the loss
		if itr // n_batch < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1 - 0.99 ** (itr // n_batch - wait_until_kl_inc))   # increasing kl_coef over iterations

		# utils.log_gpu_stats("## Before batch load ##")
		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])   # next batch of samples
		# utils.log_gpu_stats("## After batch load ##")
		train_res = model.compute_all_losses(batch_dict, n_traj = args.n_traj, kl_coef = kl_coef)
		# utils.log_gpu_stats("## After forward ##")
		train_res["loss"].backward()   # compute the gradient of current tensor wrt graph leaves w.r.t. loss
		# utils.log_gpu_stats("## After backward ##")
		optimizer.step()   # performs a single optimization step (parameter update)
		# optimizer.zero_grad()
		# utils.log_gpu_stats("## After optimizer ##")

		n_iter_to_eval = 1
		if itr % (n_iter_to_eval * n_batch) == 0:
			model.eval() # Switch to evaluation mode
			with torch.no_grad():

				test_res = utils.compute_loss_val_batches(args, model, data_obj["test_dataloader"],
						n_traj = args.n_traj, kl_coef = kl_coef)   # evaluation on the test data

				message = 'Epoch {:04d} | Loss {:.3f} | LL {:.3f} | CE Loss {:.3f} | MSE {:.3f} | AUC {:.3f} | KL {:.3f} | FP STD {:.3f} |'.format(
					itr // n_batch, test_res["loss"].detach(), test_res["likelihood"].detach(), test_res["ce_loss"].detach(),
						test_res["mse"].detach(), test_res["auc"], test_res["kl_div"], test_res["fp_std"])
			
				print(message)

		model.train()  # switch back to training mode

			# torch.cuda.empty_cache()


	# # Create the memory snapshot file
	# export_memory_snapshot()
	# # Stop recording memory snapshot history
	# stop_record_memory_history()

	print("--- Training took %s mins ---" % ((time.time()-start_time)/60))

	# torch.save(model.state_dict(), args.save_file)



