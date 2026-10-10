"""Small 1D models for 4 GB GPUs; input remains the frozen XYZ waveform."""
import torch
from torch import nn
from torch.nn import functional as F


def encoder():
    layers = []
    for cin, cout in [(3, 16), (16, 32), (32, 64)]:
        layers.extend([nn.Conv1d(cin, cout, 11, stride=5, padding=5), nn.ReLU()])
    return nn.Sequential(*layers)


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = encoder()
        self.classifier = nn.Linear(64, 6)

    def forward(self, x):
        return self.classifier(self.features(x).mean(dim=-1))


class CVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder()
        self.posterior = nn.Linear(64*24+6, 2*latent_dim)
        self.expand = nn.Linear(latent_dim+6, 64*24)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 11, stride=5, padding=5, output_padding=4), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 11, stride=5, padding=5, output_padding=4), nn.ReLU(),
            nn.ConvTranspose1d(16, 3, 11, stride=5, padding=5, output_padding=4))

    def decode(self, z, y):
        h = torch.cat([z, F.one_hot(y, 6).to(z.dtype)], dim=1)
        return self.decoder(self.expand(h).view(-1, 64, 24))

    def forward(self, x, y, eps=None):
        h = torch.cat([self.encoder(x).flatten(1), F.one_hot(y, 6).to(x.dtype)], dim=1)
        mu, logvar = self.posterior(h).chunk(2, dim=1)
        logvar = logvar.clamp(-12, 12)
        eps = torch.randn_like(mu) if eps is None else eps
        return self.decode(mu + (0.5*logvar).exp()*eps, y), mu, logvar
