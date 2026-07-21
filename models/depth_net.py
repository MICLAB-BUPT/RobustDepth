import pytorch_lightning as pl

from functools import partial
from models.md2.layers import disp_to_depth
from models.md2.resnet_encoder import ResnetEncoder
from models.md2.depth_decoder import DepthDecoder
from data.transforms import NormalizeDynamic
import torch


class DepthNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        encoder_meta = cfg.MODEL.DEPTH.ENCODER.VERSION.split('-')
        assert encoder_meta[0].lower() in ['resnet']

        self.encoder = ResnetEncoder(num_layers=int(encoder_meta[1]), pretrained=cfg.MODEL.DEPTH.ENCODER.PRETRAINED)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.scale_disp = partial(disp_to_depth, min_depth=cfg.MODEL.DEPTH.MIN_DEPTH, max_depth=cfg.MODEL.DEPTH.MAX_DEPTH)

        self.normalize = NormalizeDynamic(cfg)

    def forward(self, img, daytime, add_noise=False):

        x = self.normalize(img, daytime)
        x = self.encoder(x)
        if add_noise:
            for i in range(len(x)):
                x[i] =  x[i]  + torch.randn_like(x[i]) * 0.03
        encoder_feats = [feat.clone() for feat in x]
        x = self.decoder(x)
        output = {scale: self.scale_disp(disp)[0] for scale, disp in x.items()}
        
        return output, encoder_feats
