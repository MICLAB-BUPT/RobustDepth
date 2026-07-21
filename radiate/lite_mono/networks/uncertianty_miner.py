from networks.uncertainty_block import UncertaintyBlock
import torch
import torch.nn as nn


class UncertaintyMiner(nn.Module):
    def __init__(self, pseudo_num, pseudo_input_dim):
        super(UncertaintyMiner, self).__init__()

        # Initialize pseudo encoders using Conv2d + BatchNorm2d
        self.pseudo_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=pseudo_input_dim,
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    bias=True
                ),
                nn.BatchNorm2d(pseudo_input_dim)
            ) for _ in range(pseudo_num)
        ])

        # Initialize data uncertainty blocks
        self.data_uncertainty_blocks = nn.ModuleList([
            UncertaintyBlock(64, pseudo_input_dim, pseudo_num),
            # UncertaintyBlock(128, pseudo_input_dim, pseudo_num),
            UncertaintyBlock(224, pseudo_input_dim, pseudo_num),
        ])

        # Initialize pseudo decoders
        self.pseudo_decoders = nn.ModuleList([
            nn.Conv2d(pseudo_input_dim, 1, kernel_size=1)
            for _ in range(pseudo_num)
        ])
        
        self.upsample2x = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample4x = nn.UpsamplingNearest2d(scale_factor=4)

    def forward(self, x, features):
      
        
        e = []
        # features[1] = self.upsample2x(features[1])
        # features[2] = self.upsample4x(features[2])
        features[1] = self.upsample4x(features[1])
        
        for index, m in enumerate(self.pseudo_encoders):
            e.append(m(x[index][0]))
        
        for index, m in enumerate(self.data_uncertainty_blocks):
            e = m(features[index], e)
                
        # get the uncertainty 
        for index, m in enumerate(self.pseudo_decoders):
            e[index] = m(e[index])
        # e = [self.upsample4x(ps) for ps in e]
        # e = [torch.sigmoid(ps) for ps in e]
        
        return e