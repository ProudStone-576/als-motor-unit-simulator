import torch
import torch.nn as nn

class DegenerationODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, t, z):
        """
        z = [umn_deg, lmn_deg, umn_signal, lmn_signal]
        """
        return self.net(z)
