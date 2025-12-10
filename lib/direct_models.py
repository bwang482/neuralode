###
# Direct prediction models (non-VAE):
# - ODE_RNN: ODE-RNN encoder with direct classification from hidden states
# - Classic_RNN: Standard RNN with direct classification from hidden states
#
# These models do NOT use:
# - Latent space sampling (no z0 ~ N(mu, sigma))
# - Decoder / reconstruction
# - KL divergence loss
#
# Loss function: Classification CE only (+ optional MSE for next-step prediction)
###

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.model_functs as utils
from lib.likelihood import compute_binary_loss

# -----


class ODE_RNN(nn.Module):
    """
    ODE-RNN model for direct classification.
    
    This model uses the ODE-RNN encoder from the Latent ODE paper,
    but instead of sampling from a latent distribution, it directly
    uses the hidden states for classification.
    
    Architecture:
        Observations → ODE-RNN Encoder → Hidden State → Classifier → Prediction
    
    Loss: Classification cross-entropy only (no reconstruction, no KL)
    """

    def __init__(self, encoder, hidden_dim, n_label=1, n_unit=100, 
                 dropout_rate=0.0, use_last_hidden=True):
        """
        Args:
            encoder: ODE-RNN encoder (Encoder_z0_ODE_RNN)
            hidden_dim: Dimension of hidden state from encoder
            n_label: Number of output labels for classification
            n_unit: Number of units in classifier hidden layers
            dropout_rate: Dropout rate in classifier
            use_last_hidden: If True, use last hidden state; else use output of transform_z0
        """
        super(ODE_RNN, self).__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.n_label = n_label
        self.use_last_hidden = use_last_hidden

        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_unit),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_unit, n_unit),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_unit, n_label)
        )
        utils.init_netw_weights(self.classifier)

    def forward(self, obs_data, obs_tps, mask=None):
        """
        Forward pass for classification.
        
        Args:
            obs_data: Observed data, shape: (n_subj, n_tp, n_dim)
            obs_tps: Observation time points
            mask: Observation mask, shape: (n_subj, n_tp, n_dim)
            
        Returns:
            label_pred: Classification logits, shape: (n_subj, n_label)
            hidden: Hidden state used for classification
        """
        # Concatenate data with mask
        data_w_mask = obs_data
        if mask is not None:
            data_w_mask = torch.cat((obs_data, mask), -1)
        
        # Get hidden representation from encoder
        # The encoder outputs (mean_z0, std_z0) but we use mean_z0 directly (no sampling)
        mean_z0, std_z0 = self.encoder(data_w_mask, obs_tps)  # 1 x n_subj x hidden_dim
        
        # Use mean directly (no sampling from distribution)
        hidden = mean_z0.squeeze(0)  # n_subj x hidden_dim
        
        # Classify
        label_pred = self.classifier(hidden)  # n_subj x n_label
        
        return label_pred, hidden

    def compute_all_losses(self, batch_dict, n_traj=1, kl_coef=1., ce_weight=100, 
                          use_joint_loss=True, z_last=False):
        """
        Compute losses - only classification CE for this model.
        
        Note: n_traj, kl_coef, use_joint_loss, z_last are ignored (kept for API compatibility)
        """
        label_pred, hidden = self.forward(
            batch_dict["obs_data"],
            batch_dict["obs_tps"],
            mask=batch_dict["obs_mask"]
        )
        
        # Compute classification loss
        if batch_dict["labels"] is not None:
            # Expand label_pred to match expected shape (1, n_subj) for compute_binary_loss
            label_pred_expanded = label_pred.squeeze(-1).unsqueeze(0)  # 1 x n_subj
            ce_loss = compute_binary_loss(label_pred_expanded, batch_dict["labels"])
        else:
            ce_loss = torch.tensor(0., device=label_pred.device)
        
        # For ODE-RNN, loss is just CE (no reconstruction, no KL)
        loss = ce_loss
        
        metrics = {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "likelihood": torch.tensor(0.),  # No reconstruction
            "mse": torch.tensor(0.),         # No reconstruction
            "kl_div": torch.tensor(0.),      # No KL (not a VAE)
            "fp_std": torch.tensor(0.),      # No latent std
        }
        
        if batch_dict["labels"] is not None:
            metrics["label_pred"] = label_pred_expanded.detach()
        
        return metrics

    def compute_classification_loss_only(self, batch_dict, n_traj=1, z_last=False):
        """
        Same as compute_all_losses for this model (no reconstruction anyway).
        """
        return self.compute_all_losses(batch_dict, n_traj=n_traj, z_last=z_last)


class Classic_RNN(nn.Module):
    """
    Classic RNN model for direct classification.
    
    This is a simple GRU-based model that processes observations
    and uses the final hidden state for classification.
    
    Architecture:
        Observations → GRU → Hidden State → Classifier → Prediction
    
    Loss: Classification cross-entropy only
    """

    def __init__(self, input_dim, hidden_dim, n_label=1, n_unit=100,
                 dropout_rate=0.0, n_layers=1, bidirectional=False):
        """
        Args:
            input_dim: Input feature dimension (with mask: input_dim * 2)
            hidden_dim: Hidden dimension of GRU
            n_label: Number of output labels
            n_unit: Number of units in classifier hidden layers
            dropout_rate: Dropout rate
            n_layers: Number of GRU layers
            bidirectional: Whether to use bidirectional GRU
        """
        super(Classic_RNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_label = n_label
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if n_layers > 1 else 0.0
        )
        
        # Classifier input dimension depends on bidirectional
        classifier_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, n_unit),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_unit, n_unit),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_unit, n_label)
        )
        utils.init_netw_weights(self.classifier)

    def forward(self, obs_data, obs_tps, mask=None):
        """
        Forward pass for classification.
        
        Args:
            obs_data: Observed data, shape: (n_subj, n_tp, n_dim)
            obs_tps: Observation time points (not used, kept for API compatibility)
            mask: Observation mask, shape: (n_subj, n_tp, n_dim)
            
        Returns:
            label_pred: Classification logits, shape: (n_subj, n_label)
            hidden: Final hidden state
        """
        # Concatenate data with mask
        data_w_mask = obs_data
        if mask is not None:
            data_w_mask = torch.cat((obs_data, mask), -1)
        
        # Process through GRU (backwards in time to match other encoders)
        # Flip sequence to process from last to first timepoint
        data_reversed = torch.flip(data_w_mask, dims=[1])
        
        # GRU forward pass
        _, hidden = self.gru(data_reversed)  # hidden: (n_layers * n_directions, n_subj, hidden_dim)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden_fwd = hidden[-2, :, :]  # Last layer, forward
            hidden_bwd = hidden[-1, :, :]  # Last layer, backward
            hidden_final = torch.cat([hidden_fwd, hidden_bwd], dim=-1)  # n_subj x (hidden_dim * 2)
        else:
            hidden_final = hidden[-1, :, :]  # n_subj x hidden_dim
        
        # Classify
        label_pred = self.classifier(hidden_final)  # n_subj x n_label
        
        return label_pred, hidden_final

    def compute_all_losses(self, batch_dict, n_traj=1, kl_coef=1., ce_weight=100,
                          use_joint_loss=True, z_last=False):
        """
        Compute losses - only classification CE for this model.
        """
        label_pred, hidden = self.forward(
            batch_dict["obs_data"],
            batch_dict["obs_tps"],
            mask=batch_dict["obs_mask"]
        )
        
        # Compute classification loss
        if batch_dict["labels"] is not None:
            label_pred_expanded = label_pred.squeeze(-1).unsqueeze(0)  # 1 x n_subj
            ce_loss = compute_binary_loss(label_pred_expanded, batch_dict["labels"])
        else:
            ce_loss = torch.tensor(0., device=label_pred.device)
        
        loss = ce_loss
        
        metrics = {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "likelihood": torch.tensor(0.),
            "mse": torch.tensor(0.),
            "kl_div": torch.tensor(0.),
            "fp_std": torch.tensor(0.),
        }
        
        if batch_dict["labels"] is not None:
            metrics["label_pred"] = label_pred_expanded.detach()
        
        return metrics

    def compute_classification_loss_only(self, batch_dict, n_traj=1, z_last=False):
        """
        Same as compute_all_losses for this model.
        """
        return self.compute_all_losses(batch_dict, n_traj=n_traj, z_last=z_last)


class ODE_RNN_Attention(nn.Module):
    """
    ODE-RNN with attention over the trajectory for classification.
    
    Instead of using just the final hidden state, this model uses
    attention to aggregate information across all time points.
    """

    def __init__(self, encoder, hidden_dim, n_label=1, n_unit=100, dropout_rate=0.0):
        """
        Args:
            encoder: ODE-RNN encoder
            hidden_dim: Dimension of hidden states
            n_label: Number of output labels
            n_unit: Number of units in classifier
            dropout_rate: Dropout rate
        """
        super(ODE_RNN_Attention, self).__init__()

        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.n_label = n_label

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, n_unit),
            nn.Tanh(),
            nn.Linear(n_unit, 1)
        )
        utils.init_netw_weights(self.attention)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, n_unit),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_unit, n_unit),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_unit, n_label)
        )
        utils.init_netw_weights(self.classifier)

    def forward(self, obs_data, obs_tps, mask=None):
        """
        Forward pass with attention over encoder trajectory.
        """
        data_w_mask = obs_data
        if mask is not None:
            data_w_mask = torch.cat((obs_data, mask), -1)
        
        # Get trajectory from encoder (need to access internal method)
        # For now, use final hidden state
        mean_z0, std_z0 = self.encoder(data_w_mask, obs_tps)
        hidden = mean_z0.squeeze(0)
        
        label_pred = self.classifier(hidden)
        
        return label_pred, hidden

    def compute_all_losses(self, batch_dict, n_traj=1, kl_coef=1., ce_weight=100,
                          use_joint_loss=True, z_last=False):
        """Compute CE loss only."""
        label_pred, hidden = self.forward(
            batch_dict["obs_data"],
            batch_dict["obs_tps"],
            mask=batch_dict["obs_mask"]
        )
        
        if batch_dict["labels"] is not None:
            label_pred_expanded = label_pred.squeeze(-1).unsqueeze(0)
            ce_loss = compute_binary_loss(label_pred_expanded, batch_dict["labels"])
        else:
            ce_loss = torch.tensor(0., device=label_pred.device)
        
        metrics = {
            "loss": ce_loss,
            "ce_loss": ce_loss.detach(),
            "likelihood": torch.tensor(0.),
            "mse": torch.tensor(0.),
            "kl_div": torch.tensor(0.),
            "fp_std": torch.tensor(0.),
        }
        
        if batch_dict["labels"] is not None:
            metrics["label_pred"] = label_pred_expanded.detach()
        
        return metrics

    def compute_classification_loss_only(self, batch_dict, n_traj=1, z_last=False):
        return self.compute_all_losses(batch_dict, n_traj=n_traj, z_last=z_last)
