###

import time
import numpy as np

import torch
import torch.nn as nn

import utils.model_functs as utils
from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import *
from lib.latent_ode import LatentODE

# -----

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, n_label):

	rec_input_dim = int(input_dim) * 2   # mask concatenated
	rec_latent_dim = args.rec_latent_dims

	gen_input_dim = int(input_dim)
	gen_latent_dim = args.gen_latent_dims


	# recognition/encoding ODE

	ode_func_netw = utils.create_netw(n_input = rec_latent_dim, n_output = rec_latent_dim, n_layer = args.rec_layers, n_unit = args.units, nonlinear = nn.Tanh)
        # create a feedforward neural network mapping rec_latent_dim -> rec_latent_dim with rec_layers, each has n_unit neurons

	rec_ode_func = ODEFunc(ode_func_netw = ode_func_netw)
        # initialize the recognition ODE network
        # ode_func_net: rec_latent_dim -> rec_latent_dim

	z0_diffeq_solver = DiffeqSolver(ode_func = rec_ode_func, method = "euler", odeint_rtol = 1e-3, odeint_atol = 1e-4)
	# see diffeq_solver.py
	# initialize ODE solver of the recognition model
		
	encoder_z0 = Encoder_z0_ODE_RNN(input_dim = rec_input_dim, latent_dim = rec_latent_dim, z0_diffeq_solver = z0_diffeq_solver,
					z0_dim = gen_latent_dim, n_gru_unit = args.gru_units)
	# see encoder_decoder.py
	# calculate z0


	# generative/decoding ODE

	ode_func_netw = utils.create_netw(n_input = gen_latent_dim, n_output = gen_latent_dim, n_layer = args.gen_layers, n_unit = args.units, nonlinear = nn.Tanh)
	# create a feedforward neural network mapping gen_latent_dim -> gen_latent_dim with gen_layers, each has n_unit neurons

	gen_ode_func = ODEFunc(ode_func_netw = ode_func_netw)
	# initialize the generative ODE network
	# ode_func_net: gen_latent_dim -> gen_latent_dim

	diffeq_solver = DiffeqSolver(ode_func = gen_ode_func, method = 'dopri5', odeint_rtol = 1e-3, odeint_atol = 1e-4)
	# see diffeq_solver.py
	# initialize ODE solver of the generative model

	decoder = Decoder(gen_latent_dim, gen_input_dim)
	# see encoder_decoder.py
	# decode the latent states to the data space

	model = LatentODE(
		encoder_z0 = encoder_z0, 
		decoder = decoder, 
		diffeq_solver = diffeq_solver,
		z0_dim = gen_latent_dim,
		z0_prior = z0_prior,
		obsrv_std = obsrv_std,
		n_label = n_label,
		n_unit = args.classif_units,
		classif = args.classif,
		classif_w_recon = args.classif_w_recon)

	return model



