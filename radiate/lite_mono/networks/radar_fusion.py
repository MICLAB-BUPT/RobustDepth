import torch
import torch.nn as nn
from torchvision import models
import networks



def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
class Conv1x1Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1Adapter, self).__init__()
        # Define a 1x1 convolution to change the channel dimension
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv1x1(x)

class RadarFusion(nn.Module):
    def __init__(self, mono_c, radar_c):
        super(RadarFusion, self).__init__()
        self.encoder = networks.RadarResnetEncoder(num_layers=18, pretrained=False, num_input_images=1, n_img_channels=1)
        
        self.mono_c = mono_c
        self.radar_c = radar_c
        
        self.conv_fusion = nn.Conv2d(mono_c+radar_c, mono_c, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(mono_c)
        
        self.conv_fusion.apply(weights_init_kaiming)
        self.bn_fusion.apply(weights_init_kaiming)
        

    def forward(self, radar_input, mono_feats):
        radar_feats = self.encoder(radar_input)[-2]  # Corrected here
        x_fused = torch.cat([mono_feats, radar_feats], dim=1) 
        
        x_fused = self.bn_fusion(self.conv_fusion(x_fused))
        
        return x_fused



# --------------multi-layer sparse fusion-----------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation='relu'):
        super().__init__()

        padding = kernel_size//2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=False)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'elu':
            self.act = nn.ELU()

        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        

    def forward(self, x, mask):
        x = x*mask
        x = self.conv(x)
        normalizer = 1/(self.sparsity(mask)+1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.act(x)
        
        mask = self.max_pool(mask)

        return x, mask

# Define the encoder_radar_sparse_conv model
class RadarSMFusion(nn.Module):
    # sparse and multi-layer fusion
    def __init__(self, radar_input_channels=1, num_layers=18):
        super(RadarSMFusion, self).__init__()

        self.sparse_conv1 = SparseConv(radar_input_channels, 16, 7, activation='elu')
        self.sparse_conv2 = SparseConv(16, 16, 5, activation='elu')
        self.sparse_conv3 = SparseConv(16, 16, 3, activation='elu')
        self.sparse_conv4 = SparseConv(16, 3, 3, activation='elu')

        if num_layers == 34:
            self.base_model_radar = models.resnet34(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        elif num_layers == 18:
            self.base_model_radar = models.resnet18(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError('Not supported encoder: {}'.format(num_layers))
        
        self.adapter = Conv1x1Adapter(256, 224)
        
        num_in = [64, 128, 256]
        num_out = [64, 128, 224]
        self.weights = nn.ModuleList() 
        self.projects = nn.ModuleList() 
        for i in range(len(num_in)):
            self.weights.append(
                torch.nn.Sequential(nn.Conv2d(num_in[i], num_out[i], 1, 1, bias=False),
                nn.Sigmoid())
            )
            self.projects.append(
                torch.nn.Sequential(nn.Conv2d(num_in[i], num_out[i], 1, 1, bias=False),
                nn.ReLU())
            )
            

    def forward(self, radar_inputs, mono_feats):
        x = radar_inputs
        mask = (x[:, 0] > 0).float().unsqueeze(1)
        feature = x
        feature, mask = self.sparse_conv1(feature, mask)
        feature, mask = self.sparse_conv2(feature, mask)
        feature, mask = self.sparse_conv3(feature, mask)
        feature, mask = self.sparse_conv4(feature, mask)

        skip_feat = []
        for k, v in self.base_model_radar._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
                
        fused_feats = []
        for i in range(len(mono_feats)):
            radar_weighted = self.weights[i](skip_feat[i+1])
            radar_projected = self.projects[i](skip_feat[i+1])
            fused_result = radar_weighted * radar_projected + mono_feats[i]
            
            fused_feats.append(fused_result)
                
        
        return fused_feats
