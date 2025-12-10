###
# RNN-VAE: Variational Autoencoder with RNN encoder and RNN decoder
# No ODEs are used in this model.
###

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.model_functs as utils
from lib.base_vae import VAE_Baseline

# -----


class RNN_VAE(VAE_Baseline):
    """
    RNN-VAE: Variational Autoencoder with RNN encoder and RNN decoder.
    
    This model encodes observations using an RNN running backwards in time,
    samples a latent state z0, and decodes using another RNN to reconstruct
    the observations.
    
    Unlike Latent ODE, this model does not use differential equations.
    """

    def __init__(self, encoder_z0, decoder, z0_dim, z0_prior, obsrv_std,
                 dropout_rate=0.0, n_label=1, n_unit=100, classif=True, 
                 classif_w_recon=True, use_traj_attention=False):
        """
        Args:
            encoder_z0: RNN encoder that produces z0 mean and std
            decoder: RNN decoder that generates predictions from z0
            z0_dim: Dimension of the latent state z0
            z0_prior: Prior distribution over z0
            obsrv_std: Observation noise standard deviation
            dropout_rate: Dropout rate for classifier
            n_label: Number of labels for classification
            n_unit: Number of units in classification network
            classif: Whether to include classification
            classif_w_recon: Whether to jointly train classifier with reconstruction
            use_traj_attention: Whether to use attention over trajectory for classification
        """
        super(RNN_VAE, self).__init__(
            z0_dim=z0_dim,
            z0_prior=z0_prior,
            obsrv_std=obsrv_std,
            dropout_rate=dropout_rate,
            n_label=n_label,
            n_unit=n_unit,
            classif=classif,
            classif_w_recon=classif_w_recon,
            use_traj_attention=use_traj_attention
        )

        self.encoder_z0 = encoder_z0
        self.decoder = decoder
        self.z0_dim = z0_dim

    def get_recons(self, tps_to_pred, obs_data, obs_tps, mask, n_traj=1, z_last=False):
        """
        Get reconstructions of the observations.
        
        Args:
            tps_to_pred: Time points at which to make predictions
            obs_data: Observed data, shape: (n_subj, n_tp, n_dim)
            obs_tps: Observation time points
            mask: Observation mask, shape: (n_subj, n_tp, n_dim)
            n_traj: Number of latent trajectories to sample
            z_last: Whether to use last latent state for classification
            
        Returns:
            pred_y: Predicted observations, shape: (n_traj, n_subj, n_tp, n_dim)
            info: Dictionary with additional information
        """
        # Concatenate data with mask for encoder
        data_w_mask = obs_data
        if mask is not None:
            data_w_mask = torch.cat((obs_data, mask), -1)
        
        # Encode to get z0 distribution parameters
        fp_mu, fp_std = self.encoder_z0(data_w_mask, obs_tps)  # 1 x n_subj x z0_dim
        
        # Sample z0
        mean_z0 = fp_mu.repeat(n_traj, 1, 1)  # n_traj x n_subj x z0_dim
        std_z0 = fp_std.repeat(n_traj, 1, 1)  # n_traj x n_subj x z0_dim
        fp_enc = utils.sample_standard_gaussian(mean_z0, std_z0)  # n_traj x n_subj x z0_dim
        
        # Decode using RNN decoder
        # The decoder takes z0 and time points, returns predictions
        pred_y = self.decoder(fp_enc, tps_to_pred)  # n_traj x n_subj x n_tp x n_dim
        
        info = {
            "fp": (fp_mu, fp_std, fp_enc),
            "latent_traj": None  # No continuous latent trajectory in RNN-VAE
        }
        
        # Classification
        if self.classif:
            if self.use_traj_attention and hasattr(self, 'traj_attention'):
                # For RNN-VAE, we don't have a latent trajectory like ODE models
                # We can use the decoder hidden states or just use z0
                # For simplicity, use z0 (initial encoding)
                info["label_pred"] = self.classifier(fp_enc).squeeze(-1)
            elif z_last:
                # Use last decoder hidden state if available
                # For now, fall back to z0
                info["label_pred"] = self.classifier(fp_enc).squeeze(-1)
            else:
                info["label_pred"] = self.classifier(fp_enc).squeeze(-1)
        
        return pred_y, info

    def sample_traj_from_prior(self, tps_to_pred, n_traj=1):
        """
        Sample trajectories from the prior distribution.
        
        Args:
            tps_to_pred: Time points at which to generate predictions
            n_traj: Number of trajectories to sample
            
        Returns:
            Sampled trajectories, shape: (n_traj, 1, n_tp, n_dim)
        """
        device = next(self.parameters()).device
        
        # Sample z0 from prior
        z0 = self.z0_prior.sample([n_traj, 1, self.z0_dim]).squeeze(-1)
        z0 = z0.to(device)
        
        # Decode
        pred_y = self.decoder(z0, tps_to_pred)
        
        return pred_y

    def get_classification_only(self, tps_to_pred, obs_data, obs_tps, mask, n_traj=1, z_last=False):
        """
        Classification-only inference - minimal computation for test time.
        
        Args:
            tps_to_pred: Time points (not used for classification)
            obs_data: Observed data
            obs_tps: Observation time points
            mask: Observation mask
            n_traj: Number of trajectories to sample
            z_last: Whether to use last state (not applicable for RNN-VAE)
            
        Returns:
            None (no predictions), info dict with classification results
        """
        # Concatenate data with mask
        data_w_mask = obs_data
        if mask is not None:
            data_w_mask = torch.cat((obs_data, mask), -1)
        
        # Encode
        fp_mu, fp_std = self.encoder_z0(data_w_mask, obs_tps)
        
        # Sample z0
        mean_z0 = fp_mu.repeat(n_traj, 1, 1)
        std_z0 = fp_std.repeat(n_traj, 1, 1)
        fp_enc = utils.sample_standard_gaussian(mean_z0, std_z0)
        
        info = {
            "fp": (fp_mu, fp_std, fp_enc),
            "latent_traj": None
        }
        
        # Classification only - no decoding
        if self.classif:
            info["label_pred"] = self.classifier(fp_enc).squeeze(-1)
        
        return None, info


class RNN_VAE_Bidirectional(VAE_Baseline):
    """
    RNN-VAE with bidirectional encoder.
    
    Uses a bidirectional GRU encoder for potentially better encoding
    of the observation sequence.
    """

    def __init__(self, encoder_z0, decoder, z0_dim, z0_prior, obsrv_std,
                 dropout_rate=0.0, n_label=1, n_unit=100, classif=True, 
                 classif_w_recon=True, use_traj_attention=False):
        """
        Same as RNN_VAE but expects a bidirectional encoder.
        """
        super(RNN_VAE_Bidirectional, self).__init__(
            z0_dim=z0_dim,
            z0_prior=z0_prior,
            obsrv_std=obsrv_std,
            dropout_rate=dropout_rate,
            n_label=n_label,
            n_unit=n_unit,
            classif=classif,
            classif_w_recon=classif_w_recon,
            use_traj_attention=use_traj_attention
        )

        self.encoder_z0 = encoder_z0
        self.decoder = decoder
        self.z0_dim = z0_dim

    def get_recons(self, tps_to_pred, obs_data, obs_tps, mask, n_traj=1, z_last=False):
        """Get reconstructions - same as RNN_VAE."""
        data_w_mask = obs_data
        if mask is not None:
            data_w_mask = torch.cat((obs_data, mask), -1)
        
        fp_mu, fp_std = self.encoder_z0(data_w_mask, obs_tps)
        
        mean_z0 = fp_mu.repeat(n_traj, 1, 1)
        std_z0 = fp_std.repeat(n_traj, 1, 1)
        fp_enc = utils.sample_standard_gaussian(mean_z0, std_z0)
        
        pred_y = self.decoder(fp_enc, tps_to_pred)
        
        info = {
            "fp": (fp_mu, fp_std, fp_enc),
            "latent_traj": None
        }
        
        if self.classif:
            info["label_pred"] = self.classifier(fp_enc).squeeze(-1)
        
        return pred_y, info

    def sample_traj_from_prior(self, tps_to_pred, n_traj=1):
        """Sample trajectories from prior."""
        device = next(self.parameters()).device
        
        z0 = self.z0_prior.sample([n_traj, 1, self.z0_dim]).squeeze(-1)
        z0 = z0.to(device)
        
        pred_y = self.decoder(z0, tps_to_pred)
        
        return pred_y

    def get_classification_only(self, tps_to_pred, obs_data, obs_tps, mask, n_traj=1, z_last=False):
        """Classification-only inference."""
        data_w_mask = obs_data
        if mask is not None:
            data_w_mask = torch.cat((obs_data, mask), -1)
        
        fp_mu, fp_std = self.encoder_z0(data_w_mask, obs_tps)
        
        mean_z0 = fp_mu.repeat(n_traj, 1, 1)
        std_z0 = fp_std.repeat(n_traj, 1, 1)
        fp_enc = utils.sample_standard_gaussian(mean_z0, std_z0)
        
        info = {
            "fp": (fp_mu, fp_std, fp_enc),
            "latent_traj": None
        }
        
        if self.classif:
            info["label_pred"] = self.classifier(fp_enc).squeeze(-1)
        
        return None, info
