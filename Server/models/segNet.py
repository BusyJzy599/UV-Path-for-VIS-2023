import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .basic_block import *
from .swin_transformer import *
from .uperNet import *

pretrain_dir="/data/chenzhongming/hubmap/hubmap256/pretrain_param"
image_size=1024

cfg = dict(

        #configs/_base_/models/upernet_swin.py
        basic = dict(
            swin=dict(
                embed_dim=image_size//8,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False
            ),
            upernet=dict(
                in_channels=[image_size//8, image_size//4, image_size//2, image_size],
            ),

        ),
         #configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py
        swin_tiny_patch4_window7_224=dict(
            checkpoint = pretrain_dir+'/swin_tiny_patch4_window7_224_22k.pth',

            swin = dict(
                embed_dim=image_size//8,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
            ),
            upernet=dict(
                in_channels=[image_size//8, image_size//4, image_size//2, image_size],
            ),
        ),

      
    )



class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
def criterion_aux_loss(logit, mask):
    mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
    loss = F.binary_cross_entropy_with_logits(logit,mask)
    return loss


class Net(nn.Module):

    def load_pretrain( self,):

        checkpoint = cfg[self.arch]['checkpoint']
        print('loading %s ...'%checkpoint)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)['model']

        print(self.encoder.load_state_dict(checkpoint,strict=False))  #True


    def __init__( self,):
        super(Net, self).__init__()
        self.output_type = ['inference', 'loss']

        self.rgb = RGB()
        self.arch = 'swin_tiny_patch4_window7_224'

        self.encoder = SwinTransformerV1(
            ** {**cfg['basic']['swin'], 
                # **cfg[self.arch]['swin'],
                **{'out_norm' : LayerNorm2d} }
        )
        # encoder_dim =cfg[self.arch]['upernet']['in_channels']
        encoder_dim =cfg['basic']['upernet']['in_channels']
        

        self.decoder = UPerDecoder(
            in_dim=encoder_dim,
            ppm_pool_scale=[1, 2, 3, 6],
            ppm_dim=512,
            fpn_out_dim=256
        )

        self.logit = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.aux = nn.ModuleList([
            nn.Conv2d(256, 1, kernel_size=1, padding=0) for i in range(4)
        ])

    

    def forward(self, batch):
        x = batch['image']
        B,C,H,W = x.shape
        x = self.rgb(x)
        encoder = self.encoder(x)
        last, decoder = self.decoder(encoder)
        logit = self.logit(last)
        logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        output = {}
        if 'loss' in self.output_type:
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['mask'])
            for i in range(4):
                output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](decoder[i]),batch['mask'])

        if 'inference' in self.output_type:
            output['probability'] = torch.sigmoid(logit)
            

        return output,encoder,logit
    