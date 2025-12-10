###
# RNN Decoder for RNN-VAE
# Generates predictions at specified time points using an RNN
###

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.model_functs as utils

# -----

class Decoder_RNN(nn.Module):
    """
    RNN-based decoder that generates outputs at specified time points.
    Takes the latent state z0 and produces predictions at desired time points.
    """

    def __init__(self, latent_dim, output_dim, n_gru_unit=None, n_layers=1):
        """
        Args:
            latent_dim: Dimension of the latent state z0
            output_dim: Dimension of the output (reconstructed observations)
            n_gru_unit: Hidden dimension of the GRU (defaults to latent_dim)
            n_layers: Number of GRU layers
        """
        super(Decoder_RNN, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        if n_gru_unit is None:
            n_gru_unit = latent_dim
        self.n_gru_unit = n_gru_unit

        # Project z0 to initial hidden state
        self.z0_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, n_gru_unit),
            nn.Tanh()
        )
        utils.init_netw_weights(self.z0_to_hidden)

        # Time embedding layer (optional - helps the model understand time)
        self.time_embed_dim = 16
        self.time_embedding = nn.Linear(1, self.time_embed_dim)
        
        # GRU for generating sequence
        # Input is the previous output concatenated with time embedding
        self.gru = nn.GRU(
            input_size=output_dim + self.time_embed_dim,
            hidden_size=n_gru_unit,
            num_layers=n_layers,
            batch_first=True
        )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(n_gru_unit, n_gru_unit),
            nn.ReLU(),
            nn.Linear(n_gru_unit, output_dim)
        )
        utils.init_netw_weights(self.output_layer)

    def forward(self, z0, tps_to_pred):
        """
        Generate predictions at specified time points.
        
        Args:
            z0: Initial latent state, shape: (n_traj, n_subj, latent_dim)
            tps_to_pred: Time points to predict, shape: (n_tp,)
            
        Returns:
            pred_y: Predictions at each time point, shape: (n_traj, n_subj, n_tp, output_dim)
        """
        n_traj, n_subj, _ = z0.size()
        n_tp = len(tps_to_pred)
        device = z0.device

        # Reshape z0 for processing: (n_traj * n_subj, latent_dim)
        z0_flat = z0.reshape(n_traj * n_subj, -1)

        # Initialize hidden state from z0
        # h0: (n_layers, n_traj * n_subj, n_gru_unit)
        h0 = self.z0_to_hidden(z0_flat)
        h0 = h0.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # Initialize first input (zeros)
        current_input = torch.zeros(n_traj * n_subj, 1, self.output_dim, device=device)

        # Generate outputs at each time point
        outputs = []
        hidden = h0

        for t_idx in range(n_tp):
            # Get time embedding
            t = tps_to_pred[t_idx].view(1, 1, 1).expand(n_traj * n_subj, 1, 1)
            time_embed = self.time_embedding(t)  # (n_traj * n_subj, 1, time_embed_dim)

            # Concatenate input with time embedding
            gru_input = torch.cat([current_input, time_embed], dim=-1)

            # GRU step
            output, hidden = self.gru(gru_input, hidden)

            # Project to output dimension
            pred = self.output_layer(output)  # (n_traj * n_subj, 1, output_dim)
            outputs.append(pred)

            # Use current prediction as next input (autoregressive)
            current_input = pred

        # Stack outputs: (n_traj * n_subj, n_tp, output_dim)
        pred_y = torch.cat(outputs, dim=1)

        # Reshape back to (n_traj, n_subj, n_tp, output_dim)
        pred_y = pred_y.reshape(n_traj, n_subj, n_tp, self.output_dim)

        return pred_y


class Decoder_RNN_InputDriven(nn.Module):
    """
    Alternative RNN decoder that uses time differences as inputs.
    This version explicitly models the time intervals between predictions.
    """

    def __init__(self, latent_dim, output_dim, n_gru_unit=None, n_layers=1):
        """
        Args:
            latent_dim: Dimension of the latent state z0
            output_dim: Dimension of the output
            n_gru_unit: Hidden dimension of the GRU
            n_layers: Number of GRU layers
        """
        super(Decoder_RNN_InputDriven, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        if n_gru_unit is None:
            n_gru_unit = latent_dim
        self.n_gru_unit = n_gru_unit

        # Project z0 to initial hidden state for each layer
        self.z0_to_hidden = nn.Linear(latent_dim, n_gru_unit * n_layers)
        utils.init_netw_weights(self.z0_to_hidden)

        # Time interval embedding
        self.delta_t_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        utils.init_netw_weights(self.delta_t_embed)

        # GRU - input is previous output + time interval embedding
        self.gru = nn.GRU(
            input_size=output_dim + 32,
            hidden_size=n_gru_unit,
            num_layers=n_layers,
            batch_first=True
        )

        # Output projection
        self.output_layer = nn.Linear(n_gru_unit, output_dim)
        utils.init_netw_weights(self.output_layer)

    def forward(self, z0, tps_to_pred):
        """
        Generate predictions at specified time points.
        
        Args:
            z0: Initial latent state, shape: (n_traj, n_subj, latent_dim)
            tps_to_pred: Time points to predict, shape: (n_tp,)
            
        Returns:
            pred_y: Predictions, shape: (n_traj, n_subj, n_tp, output_dim)
        """
        n_traj, n_subj, _ = z0.size()
        n_tp = len(tps_to_pred)
        device = z0.device

        # Flatten batch dimensions
        z0_flat = z0.reshape(n_traj * n_subj, -1)

        # Initialize hidden state from z0
        h0 = self.z0_to_hidden(z0_flat)  # (n_traj * n_subj, n_gru_unit * n_layers)
        h0 = h0.reshape(n_traj * n_subj, self.n_layers, self.n_gru_unit)
        h0 = h0.permute(1, 0, 2).contiguous()  # (n_layers, n_traj * n_subj, n_gru_unit)

        # Initialize
        current_output = torch.zeros(n_traj * n_subj, 1, self.output_dim, device=device)
        prev_t = torch.zeros(1, device=device)

        outputs = []
        hidden = h0

        for t_idx in range(n_tp):
            # Compute time difference
            current_t = tps_to_pred[t_idx]
            delta_t = (current_t - prev_t).view(1, 1, 1).expand(n_traj * n_subj, 1, 1)

            # Embed time difference
            delta_t_embed = self.delta_t_embed(delta_t)

            # Concatenate previous output with time embedding
            gru_input = torch.cat([current_output, delta_t_embed], dim=-1)

            # GRU step
            output, hidden = self.gru(gru_input, hidden)

            # Project to output
            pred = self.output_layer(output)
            outputs.append(pred)

            # Update for next step
            current_output = pred
            prev_t = current_t

        # Stack and reshape
        pred_y = torch.cat(outputs, dim=1)
        pred_y = pred_y.reshape(n_traj, n_subj, n_tp, self.output_dim)

        return pred_y


class Decoder_RNN_Simple(nn.Module):
    """
    Simplified RNN decoder that generates the entire sequence at once.
    Uses z0 as the initial hidden state and runs GRU for n_tp steps.
    """

    def __init__(self, latent_dim, output_dim, n_gru_unit=None):
        """
        Args:
            latent_dim: Dimension of the latent state z0
            output_dim: Dimension of the output
            n_gru_unit: Hidden dimension of the GRU
        """
        super(Decoder_RNN_Simple, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        if n_gru_unit is None:
            n_gru_unit = latent_dim
        self.n_gru_unit = n_gru_unit

        # Project z0 to GRU hidden state
        self.z0_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, n_gru_unit),
            nn.Tanh()
        )
        utils.init_netw_weights(self.z0_to_hidden)

        # GRU cell for step-by-step generation
        self.gru_cell = nn.GRUCell(
            input_size=output_dim + 1,  # Previous output + normalized time
            hidden_size=n_gru_unit
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(n_gru_unit, output_dim)
        )
        utils.init_netw_weights(self.output_layer)

    def forward(self, z0, tps_to_pred):
        """
        Generate predictions at specified time points.
        
        Args:
            z0: Initial latent state, shape: (n_traj, n_subj, latent_dim)
            tps_to_pred: Time points to predict, shape: (n_tp,)
            
        Returns:
            pred_y: Predictions, shape: (n_traj, n_subj, n_tp, output_dim)
        """
        n_traj, n_subj, _ = z0.size()
        n_tp = len(tps_to_pred)
        device = z0.device

        # Flatten batch dimensions
        batch_size = n_traj * n_subj
        z0_flat = z0.reshape(batch_size, -1)

        # Initialize hidden state
        hidden = self.z0_to_hidden(z0_flat)  # (batch_size, n_gru_unit)

        # Initialize first input
        current_output = torch.zeros(batch_size, self.output_dim, device=device)

        outputs = []

        for t_idx in range(n_tp):
            # Get normalized time
            t = tps_to_pred[t_idx].view(1, 1).expand(batch_size, 1)

            # Concatenate previous output with time
            gru_input = torch.cat([current_output, t], dim=-1)  # (batch_size, output_dim + 1)

            # GRU cell step
            hidden = self.gru_cell(gru_input, hidden)

            # Generate output
            pred = self.output_layer(hidden)  # (batch_size, output_dim)
            outputs.append(pred.unsqueeze(1))

            # Update for next step
            current_output = pred

        # Stack outputs: (batch_size, n_tp, output_dim)
        pred_y = torch.cat(outputs, dim=1)

        # Reshape to (n_traj, n_subj, n_tp, output_dim)
        pred_y = pred_y.reshape(n_traj, n_subj, n_tp, self.output_dim)

        return pred_y
