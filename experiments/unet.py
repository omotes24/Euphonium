# unet.py -- EDM-preconditioned U-Net (NVlabs SongUNet / DDPM++ blocks) used for the models trained in this project
import os, sys
import torch

ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f'{ROOT}/ext/edm')
from training.networks import SongUNet  # noqa: E402

PRESETS = {
    # name: (model_channels, channel_mult, num_blocks, attn_resolutions)
    's':  (96,  [1, 2, 2, 2], 2, [16]),
    'm':  (128, [1, 2, 2, 2], 2, [16]),
    'l':  (128, [1, 2, 2, 2], 3, [16]),
    'xl': (128, [2, 2, 2],    4, [16]),      # EDM DDPM++ CIFAR-10 configuration
}


class Net(torch.nn.Module):
    """D(x_noisy, sigma) = c_skip * x + c_out * F(c_in * x, ln(sigma)/4)   (Karras et al. 2022)."""
    def __init__(self, cfg):
        super().__init__()
        mc, mult, nb, attn = PRESETS[cfg['arch']]
        self.cfg = dict(cfg)
        self.sigma_data = float(cfg.get('sigma_data', 0.5))
        self.fp16 = bool(cfg.get('fp16', True))
        self.unet = SongUNet(img_resolution=cfg.get('res', 32), in_channels=cfg.get('ch', 3), out_channels=cfg.get('ch', 3),
                             label_dim=cfg.get('label_dim', 0), augment_dim=0, model_channels=mc, channel_mult=mult,
                             channel_mult_emb=4, num_blocks=nb, attn_resolutions=attn, dropout=cfg.get('dropout', 0.13),
                             label_dropout=0, embedding_type='positional', channel_mult_noise=1,
                             encoder_type='standard', decoder_type='standard', resample_filter=[1, 1])

    def forward(self, x, sigma, labels=None):
        s = sigma.reshape(-1, 1, 1, 1).float()
        sd = self.sigma_data
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + s ** 2).sqrt()
        c_noise = s.log() / 4
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.fp16):
            F = self.unet(c_in * x, c_noise.flatten(), class_labels=labels, augment_labels=None)
        return c_skip * x + c_out * F.float()


def build_net(cfg):
    return Net(cfg)
