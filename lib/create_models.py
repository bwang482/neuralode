###
# Factory functions for creating all model variants
# 
# VAE-based models (encoder → z0 sampling → decoder):
#   1. latent_ode: Original Latent ODE with ODE-RNN encoder + ODE decoder
#   2. latent_ode_rnn: Latent ODE with RNN encoder + ODE decoder
#   3. rnn_vae: RNN encoder + RNN decoder (no ODEs)
#
# Direct prediction models (encoder → hidden → classifier):
#   4. ode_rnn: ODE-RNN encoder, direct classification (no VAE)
#   5. classic_rnn: Standard RNN, direct classification (no VAE, no ODE)
###

import torch
import torch.nn as nn

import utils.model_functs as utils
from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import Encoder_z0_ODE_RNN, Decoder
from lib.latent_ode import LatentODE
from lib.encoder_z0_rnn import Encoder_z0_RNN, Encoder_z0_RNN_Attention
from lib.decoder_rnn import Decoder_RNN, Decoder_RNN_Simple, Decoder_RNN_InputDriven
from lib.rnn_vae import RNN_VAE, RNN_VAE_Bidirectional
from lib.direct_models import ODE_RNN, Classic_RNN

# Import existing factory function for original model
from lib.create_latent_ode_model import create_LatentODE_model

# -----


def create_LatentODE_RNN_encoder_model(args, input_dim, z0_prior, obsrv_std, n_label):
    """
    Create Latent ODE model with standard RNN encoder.
    
    This variant uses:
    - RNN (GRU) encoder: Standard GRU for encoding (no ODE between time points)
    - ODE decoder: Neural ODE for generating latent trajectories
    - Linear decoder: Maps latent states to observations
    
    Args:
        args: Command line arguments containing model hyperparameters
        input_dim: Input feature dimension
        z0_prior: Prior distribution over initial latent state
        obsrv_std: Observation noise standard deviation
        n_label: Number of output labels for classification
        
    Returns:
        model: LatentODE model with RNN encoder
    """
    rec_input_dim = int(input_dim) * 2  # mask concatenated
    rec_latent_dim = args.rec_latent_dims

    gen_input_dim = int(input_dim)
    gen_latent_dim = args.gen_latent_dims

    # ========== RNN Encoder ==========
    use_attention = getattr(args, 'use_attention', False)
    bidirectional = getattr(args, 'bidirectional', False)
    
    if use_attention:
        encoder_z0 = Encoder_z0_RNN_Attention(
            input_dim=rec_input_dim,
            latent_dim=rec_latent_dim,
            z0_dim=gen_latent_dim,
            n_gru_unit=args.gru_units
        )
    else:
        encoder_z0 = Encoder_z0_RNN(
            input_dim=rec_input_dim,
            latent_dim=rec_latent_dim,
            z0_dim=gen_latent_dim,
            n_gru_unit=args.gru_units,
            bidirectional=bidirectional
        )

    # ========== Generative/Decoding ODE ==========
    ode_func_netw = utils.create_netw(
        n_input=gen_latent_dim,
        n_output=gen_latent_dim,
        n_layer=args.gen_layers,
        n_unit=args.units,
        nonlinear=nn.Tanh
    )

    gen_ode_func = ODEFunc(ode_func_netw=ode_func_netw)

    diffeq_solver = DiffeqSolver(
        ode_func=gen_ode_func,
        method='dopri5',
        odeint_rtol=1e-3,
        odeint_atol=1e-4
    )

    decoder = Decoder(gen_latent_dim, gen_input_dim)

    # ========== Create Model ==========
    model = LatentODE(
        encoder_z0=encoder_z0,
        decoder=decoder,
        diffeq_solver=diffeq_solver,
        z0_dim=gen_latent_dim,
        z0_prior=z0_prior,
        obsrv_std=obsrv_std,
        dropout_rate=args.dropout_rate,
        n_label=n_label,
        n_unit=args.classif_units,
        classif=args.classif,
        classif_w_recon=args.classif_w_recon,
        use_traj_attention=args.use_traj_attention
    )

    return model


def create_RNN_VAE_model(args, input_dim, z0_prior, obsrv_std, n_label):
    """
    Create RNN-VAE model (no ODEs).
    
    This model uses:
    - RNN (GRU) encoder: Standard GRU for encoding
    - RNN (GRU) decoder: Standard GRU for decoding
    
    No differential equations are used in this model.
    
    Args:
        args: Command line arguments containing model hyperparameters
        input_dim: Input feature dimension
        z0_prior: Prior distribution over initial latent state
        obsrv_std: Observation noise standard deviation
        n_label: Number of output labels for classification
        
    Returns:
        model: RNN_VAE model instance
    """
    rec_input_dim = int(input_dim) * 2  # mask concatenated
    rec_latent_dim = args.rec_latent_dims

    gen_input_dim = int(input_dim)
    gen_latent_dim = args.gen_latent_dims

    # ========== RNN Encoder ==========
    bidirectional = getattr(args, 'bidirectional', False)
    use_attention = getattr(args, 'use_attention', False)
    
    if use_attention:
        encoder_z0 = Encoder_z0_RNN_Attention(
            input_dim=rec_input_dim,
            latent_dim=rec_latent_dim,
            z0_dim=gen_latent_dim,
            n_gru_unit=args.gru_units
        )
    else:
        encoder_z0 = Encoder_z0_RNN(
            input_dim=rec_input_dim,
            latent_dim=rec_latent_dim,
            z0_dim=gen_latent_dim,
            n_gru_unit=args.gru_units,
            bidirectional=bidirectional
        )

    # ========== RNN Decoder ==========
    decoder_type = getattr(args, 'decoder_type', 'simple')
    dec_layers = getattr(args, 'dec_layers', 1)
    
    if decoder_type == 'simple':
        decoder = Decoder_RNN_Simple(
            latent_dim=gen_latent_dim,
            output_dim=gen_input_dim,
            n_gru_unit=args.gru_units
        )
    elif decoder_type == 'input_driven':
        decoder = Decoder_RNN_InputDriven(
            latent_dim=gen_latent_dim,
            output_dim=gen_input_dim,
            n_gru_unit=args.gru_units,
            n_layers=dec_layers
        )
    else:  # 'standard'
        decoder = Decoder_RNN(
            latent_dim=gen_latent_dim,
            output_dim=gen_input_dim,
            n_gru_unit=args.gru_units,
            n_layers=dec_layers
        )

    # ========== Create Model ==========
    if bidirectional:
        model = RNN_VAE_Bidirectional(
            encoder_z0=encoder_z0,
            decoder=decoder,
            z0_dim=gen_latent_dim,
            z0_prior=z0_prior,
            obsrv_std=obsrv_std,
            dropout_rate=args.dropout_rate,
            n_label=n_label,
            n_unit=args.classif_units,
            classif=args.classif,
            classif_w_recon=args.classif_w_recon,
            use_traj_attention=getattr(args, 'use_traj_attention', False)
        )
    else:
        model = RNN_VAE(
            encoder_z0=encoder_z0,
            decoder=decoder,
            z0_dim=gen_latent_dim,
            z0_prior=z0_prior,
            obsrv_std=obsrv_std,
            dropout_rate=args.dropout_rate,
            n_label=n_label,
            n_unit=args.classif_units,
            classif=args.classif,
            classif_w_recon=args.classif_w_recon,
            use_traj_attention=getattr(args, 'use_traj_attention', False)
        )

    return model


def create_ODE_RNN_model(args, input_dim, z0_prior, obsrv_std, n_label):
    """
    Create ODE-RNN model for direct classification (non-VAE).
    
    This model uses:
    - ODE-RNN encoder (same as Latent ODE)
    - Direct classification from hidden state (no sampling, no decoder)
    
    Loss: Classification CE only (no reconstruction, no KL)
    
    Args:
        args: Command line arguments
        input_dim: Input feature dimension
        z0_prior: Not used (kept for API compatibility)
        obsrv_std: Not used (kept for API compatibility)
        n_label: Number of output labels
        
    Returns:
        model: ODE_RNN model instance
    """
    rec_input_dim = int(input_dim) * 2  # mask concatenated
    rec_latent_dim = args.rec_latent_dims
    gen_latent_dim = args.gen_latent_dims

    # ========== ODE-RNN Encoder ==========
    # Same as in Latent ODE
    ode_func_netw = utils.create_netw(
        n_input=rec_latent_dim,
        n_output=rec_latent_dim,
        n_layer=args.rec_layers,
        n_unit=args.units,
        nonlinear=nn.Tanh
    )

    rec_ode_func = ODEFunc(ode_func_netw=ode_func_netw)

    z0_diffeq_solver = DiffeqSolver(
        ode_func=rec_ode_func,
        method="euler",
        odeint_rtol=1e-3,
        odeint_atol=1e-4
    )

    encoder = Encoder_z0_ODE_RNN(
        input_dim=rec_input_dim,
        latent_dim=rec_latent_dim,
        z0_diffeq_solver=z0_diffeq_solver,
        z0_dim=gen_latent_dim,
        n_gru_unit=args.gru_units
    )

    # ========== Create Model ==========
    model = ODE_RNN(
        encoder=encoder,
        hidden_dim=gen_latent_dim,
        n_label=n_label,
        n_unit=args.classif_units,
        dropout_rate=args.dropout_rate,
        use_last_hidden=True
    )

    return model


def create_Classic_RNN_model(args, input_dim, z0_prior, obsrv_std, n_label):
    """
    Create Classic RNN model for direct classification (non-VAE, non-ODE).
    
    This is the simplest baseline:
    - Standard GRU encoder
    - Direct classification from hidden state
    
    Loss: Classification CE only
    
    Args:
        args: Command line arguments
        input_dim: Input feature dimension
        z0_prior: Not used (kept for API compatibility)
        obsrv_std: Not used (kept for API compatibility)
        n_label: Number of output labels
        
    Returns:
        model: Classic_RNN model instance
    """
    rec_input_dim = int(input_dim) * 2  # mask concatenated
    hidden_dim = args.gru_units
    
    bidirectional = getattr(args, 'bidirectional', False)
    n_layers = getattr(args, 'rnn_layers', 1)

    # ========== Create Model ==========
    model = Classic_RNN(
        input_dim=rec_input_dim,
        hidden_dim=hidden_dim,
        n_label=n_label,
        n_unit=args.classif_units,
        dropout_rate=args.dropout_rate,
        n_layers=n_layers,
        bidirectional=bidirectional
    )

    return model


def create_model(args, input_dim, z0_prior, obsrv_std, n_label):
    """
    Unified factory function to create models based on model_type argument.
    
    Args:
        args: Command line arguments (must include args.model_type)
        input_dim: Input feature dimension
        z0_prior: Prior distribution over initial latent state
        obsrv_std: Observation noise standard deviation
        n_label: Number of output labels for classification
        
    Returns:
        model: Model instance of the specified type
        
    Supported model types:
    
    VAE-based (encoder → z0 → decoder, joint loss):
        - 'latent_ode': Original Latent ODE with ODE-RNN encoder
        - 'latent_ode_rnn': Latent ODE with standard RNN encoder  
        - 'rnn_vae': RNN-VAE (no ODEs)
    
    Direct prediction (encoder → classifier, CE loss only):
        - 'ode_rnn': ODE-RNN encoder, direct classification
        - 'classic_rnn': Standard RNN, direct classification
    """
    model_type = getattr(args, 'model_type', 'latent_ode')
    
    # VAE-based models
    if model_type == 'latent_ode':
        return create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, n_label)
    
    elif model_type == 'latent_ode_rnn':
        return create_LatentODE_RNN_encoder_model(args, input_dim, z0_prior, obsrv_std, n_label)
    
    elif model_type == 'rnn_vae':
        return create_RNN_VAE_model(args, input_dim, z0_prior, obsrv_std, n_label)
    
    # Direct prediction models
    elif model_type == 'ode_rnn':
        return create_ODE_RNN_model(args, input_dim, z0_prior, obsrv_std, n_label)
    
    elif model_type == 'classic_rnn':
        return create_Classic_RNN_model(args, input_dim, z0_prior, obsrv_std, n_label)
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Choose from: 'latent_ode', 'latent_ode_rnn', 'rnn_vae', 'ode_rnn', 'classic_rnn'"
        )


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model, model_type=None):
    """Print a summary of the model architecture."""
    total, trainable = count_parameters(model)
    
    print("\n" + "="*60)
    if model_type:
        print(f"Model Type: {model_type}")
        
        # Print model category
        if model_type in ['latent_ode', 'latent_ode_rnn', 'rnn_vae']:
            print("Category: VAE-based (reconstruction + KL + classification)")
        else:
            print("Category: Direct prediction (classification only)")
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print("="*60 + "\n")
