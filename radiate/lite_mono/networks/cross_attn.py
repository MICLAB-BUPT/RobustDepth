import torch
import torch.nn as nn
import math
import numpy as np

class Conv1x1Adapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1Adapter, self).__init__()
        # Define a 1x1 convolution to change the channel dimension
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv1x1(x)

class EmbedCrossAttention(nn.Module):
    '''
    taking from huggingface's self attention, and modified to the cross attn
    add the key norm, proj, remove head_mask
    '''
    def __init__(self, 
                 num_heads, 
                 hidden_size, 
                 qkv_bias,attn_drop=0.,
                 proj_drop=0.) -> None:
        super(EmbedCrossAttention, self).__init__()

        self.num_attention_heads =num_heads
        self.hidden_size =hidden_size
        self.key_hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.key_hidden_size, self.all_head_size, bias=self.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=self.qkv_bias)

        self.key_norm = nn.LayerNorm(self.key_hidden_size)
        self.attn_dropout = nn.Dropout(attn_drop)
        # self.proj_dropout = nn.Dropout(proj_drop)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, key, hidden_states, attn_mask
    ) :
        # assume the head_mask is of shape [ target_seq_len, source_seq_len]
        mixed_query_layer = self.query(self.key_norm(key))

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        
        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(attn_mask, float('-inf'))

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        output = context_layer
        return output
    
    

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
    
    
class PBCrossAttention(nn.Module):
    def __init__(self, 
                mono_height, mono_width,
                downsample_rate,
                mono_c,
                radar_c,
                bev_res=200./640,
                intrinsics=None,
                num_heads=4,
                qkv_bias=True,
                attn_drop=0.,
                use_radar_gate=False,
                ) -> None:
        super(PBCrossAttention, self).__init__()
        
        if intrinsics is None:
            intrinsics = self.get_intrinsics()
        
        intrinsics = torch.from_numpy(intrinsics)
        
        self.intrinsics = intrinsics
        self.bev_res = bev_res
        self.bev_mask = self.get_geom_feats(mono_height, mono_width, downsample_rate, downsample_rate)
        self.cross_attn = EmbedCrossAttention(num_heads, mono_c, qkv_bias, attn_drop)
        
        if mono_c != radar_c:
            self.adapter = Conv1x1Adapter(radar_c, mono_c)
        self.mono_c = mono_c
        self.radar_c = radar_c
        # self.use_radar_gate = use_radar_gate
        # if self.use_radar_gate:
        #     self.gate = torch.nn.Sequential(nn.Conv2d(mono_c, mono_c, 1, 1, bias=False),
        #                 nn.Sigmoid())
        #     self.proj = torch.nn.Sequential(nn.Conv2d(mono_c, mono_c, 1, 1, bias=False),
        #                 nn.GELU())
            
        self.conv_fusion = nn.Conv2d(mono_c*2, mono_c, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(mono_c)
        
        self.conv_fusion.apply(weights_init_kaiming)
        self.bn_fusion.apply(weights_init_kaiming)
    
    def forward(self, mono_feats, radar_feats):
        if self.mono_c != self.radar_c:
            radar_feats  = self.adapter(radar_feats)
        bp_feats = self.bev_pov_cross(self.cross_attn, self.bev_mask, mono_feats, radar_feats)
        x_fused = torch.cat([mono_feats, radar_feats], dim=1) 
        
        x_fused = self.bn_fusion(self.conv_fusion(x_fused))
        # if self.use_radar_gate:
        #     bp_feats = self.gate(bp_feats) * self.proj(bp_feats)
        return x_fused
    
        
        
    def bev_pov_cross(self, pov_bev_attn_func, bev_mask, pov_feats, bev_feats):
        bs, c, h, w = pov_feats.size()

        # Corrected bev_feats_flat
        bev_feats_flat = bev_feats.view(bs, c, -1)  # Shape: (bs, c, bevh*bevw)

        # Prepare query and key features
        pov_feats_flat = pov_feats.view(bs, c, -1)  # Shape: (bs, c, h*w)
        q_feats = pov_feats_flat.permute(0, 2, 1)   # Shape: (bs, h*w, c)
        k_feats = bev_feats_flat.permute(0, 2, 1)   # Shape: (bs, bevh*bevw, c)

        # Ensure bev_mask has the correct shape: (bs, h*w, bevh*bevw)
        attn_mask = bev_mask.view(h*w, -1).to(q_feats.device)  
        attn_mask = ~attn_mask.bool() # invert, the ones will be filled with -inf before softmax

        attn_feats = pov_bev_attn_func(q_feats, k_feats, attn_mask=attn_mask)
        attn_feats = attn_feats.permute(0, 2, 1).view(bs, c, h, w)

        return attn_feats
        
    def get_geom_feats(self, ogfH=320, ogfW=640, pov_downsample=1., bev_downsample=1.):
        '''
        create_frustum
        '''
        intrinsics = self.intrinsics.clone()
        bev_res = self.bev_res
        xbound = [-100, 0., bev_res*bev_downsample]
        ybound = [-100., 100., bev_res*bev_downsample]
        zbound = [-100, 100, 200]
        grid_config = {}
        grid_config['xbound'] = xbound
        grid_config['ybound'] = ybound
        grid_config['dbound'] = [0, 100., 1.] # it determines the overall points number
        grid_config['zbound'] = zbound
        # make grid in image plane
        dbound = grid_config['dbound']
        xbound = grid_config['xbound']
        ybound = grid_config['ybound']
        zbound = grid_config['zbound']
        

        dx, bx, nx = self.gen_dx_bx(xbound, ybound, zbound)

        fH, fW = ogfH // pov_downsample, ogfW // pov_downsample
        
        # scale to init size
        intrinsics[0, 0] *= ogfW
        intrinsics[0, 2] *= ogfW
        intrinsics[1, 1] *= ogfH
        intrinsics[1, 2] *= ogfH


        ds = torch.arange(*dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3

        '''
        get geometry points
        '''
        frustum = torch.stack((xs, ys, ds), -1)
        points = torch.cat((frustum[:, :, :, :2] * frustum[:, :, :, 2:3],
                            frustum[:, :, :, 2:3]
                            ), -1)

        cam_to_ego = torch.tensor([
            [-0., -0., -1.],
            [ 1.,  0.,  0.],
            [ 0.,  -1.,  0.]])
        rots = torch.eye(3)@cam_to_ego
        points = rots.to(intrinsics.device).matmul(torch.inverse(intrinsics)).view( 1, 1, 1, 3, 3).matmul(points.to(intrinsics.device).unsqueeze(-1)).squeeze(-1)

        # to img coords
        geom_feats = ((points - (bx - dx/2.)) / dx).long() # [ds, bevh, bevw, 3]
        
        
        # get the bev_mask 
        
        bev_mask = torch.zeros(( fH*fW, nx[0], nx[1]))
        for idx in range(fH * fW):
            i, j = idx // fW, idx % fW
            ray = geom_feats[:, i, j, :]

            kept = (ray[..., 0] >= 0) & (ray[..., 0] < nx[0])\
                & (ray[..., 1] >= 0) & (ray[..., 1] < nx[1])\
                & (ray[..., 2] >= 0) & (ray[..., 2] < nx[2])
            ray = ray[kept]

            bev_mask[idx, ray[..., 0], ray[..., 1]] = 1
            
        return nn.Parameter(bev_mask, requires_grad=False)
    
    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
        
    def get_intrinsics(self):
        # return normlized intrinsics
        intrinsics = np.eye(3, dtype=np.float32)
        
        fx = 3.379191448899105e+02
        fy = 3.386957068549526e+02
        cx = 3.417366010946575e+02
        cy = 2.007359735313929e+02

        in_h, in_w = 376, 672
        intrinsics[0, 0] = fx/in_w
        intrinsics[0, 2] = cx/in_w
        intrinsics[1, 1] = fy/in_h
        intrinsics[1, 2] = cy/in_h
        return intrinsics