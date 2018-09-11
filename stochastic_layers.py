import torch

import torch.nn as nn

from model_utils import gaussian


class gaussian_layer(nn.Module):
    """
        h
        |
        z
        |
        x
    """
    def __init__(self, input_size, latent_z_size):
        super(gaussian_layer, self).__init__()
        self.input_size = input_size
        self.latent_z_size = latent_z_size

        self.q_mean2_mlp = nn.Linear(input_size, latent_z_size)
        self.q_logvar2_mlp = nn.Linear(input_size, latent_z_size)

    def forward(self, inputs, mask, sample):
        """
        inputs: batch x batch_len x input_size
        """
        batch_size, batch_len, _ = inputs.size()
        mean_qs = self.q_mean2_mlp(inputs)
        logvar_qs = self.q_logvar2_mlp(inputs)

        if sample:
            z = gaussian(mean_qs, logvar_qs) * mask.unsqueeze(-1)
        else:
            z = mean_qs * mask.unsqueeze(-1)

        return z, mean_qs, logvar_qs


class gaussian_flat_layer(nn.Module):
    """
        h
       / \
      y   z
       \ /
        x

    """
    def __init__(self, input_size, latent_z_size, latent_y_size):
        super(gaussian_flat_layer, self).__init__()
        self.input_size = input_size
        self.latent_y_size = latent_y_size
        self.latent_z_size = latent_z_size

        self.q_mean_mlp = nn.Linear(input_size, latent_z_size)
        self.q_logvar_mlp = nn.Linear(input_size, latent_z_size)

        self.q_mean2_mlp = nn.Linear(input_size, latent_y_size)
        self.q_logvar2_mlp = nn.Linear(input_size, latent_y_size)

    def forward(self, inputs, mask, sample):
        """
        inputs: batch x batch_len x input_size
        """
        batch_size, batch_len, _ = inputs.size()

        mean_qs = self.q_mean_mlp(inputs) * mask.unsqueeze(-1)
        logvar_qs = self.q_logvar_mlp(inputs) * mask.unsqueeze(-1)

        mean2_qs = self.q_mean2_mlp(inputs) * mask.unsqueeze(-1)
        logvar2_qs = self.q_logvar2_mlp(inputs) * mask.unsqueeze(-1)

        if sample:
            y = gaussian(mean2_qs, logvar2_qs) * mask.unsqueeze(-1)
        else:
            y = mean2_qs * mask.unsqueeze(-1)

        if sample:
            z = gaussian(mean_qs, logvar_qs) * mask.unsqueeze(-1)
        else:
            z = mean_qs * mask.unsqueeze(-1)

        return z, y, mean_qs, logvar_qs, mean2_qs, logvar2_qs


class gaussian_hier_layer(nn.Module):
    """
        h
        |
        y
        |
        z
        |
        x
    """
    def __init__(self, input_size, latent_z_size, latent_y_size):
        super(gaussian_hier_layer, self).__init__()
        self.input_size = input_size
        self.latent_y_size = latent_y_size
        self.latent_z_size = latent_z_size

        self.q_mean2_mlp = nn.Linear(input_size, latent_y_size)
        self.q_logvar2_mlp = nn.Linear(input_size, latent_y_size)

        self.q_mean_mlp = nn.Linear(input_size + latent_y_size, latent_z_size)
        self.q_logvar_mlp = nn.Linear(
            input_size + latent_y_size, latent_z_size)

    def forward(self, inputs, mask, sample):
        """
        inputs: batch x batch_len x input_size
        """
        batch_size, batch_len, _ = inputs.size()

        mean2_qs = self.q_mean2_mlp(inputs)
        logvar2_qs = self.q_logvar2_mlp(inputs)

        if sample:
            y = gaussian(mean2_qs, logvar2_qs) * mask.unsqueeze(-1)
        else:
            y = mean2_qs * mask.unsqueeze(-1)

        gauss_input = torch.cat([inputs, y], -1)
        mean_qs = self.q_mean_mlp(gauss_input)
        logvar_qs = self.q_logvar_mlp(gauss_input)

        if sample:
            z = gaussian(mean_qs, logvar_qs) * mask.unsqueeze(-1)
        else:
            z = mean_qs * mask.unsqueeze(-1)

        return z, y, mean_qs, logvar_qs, mean2_qs, logvar2_qs
