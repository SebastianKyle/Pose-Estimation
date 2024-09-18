import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import scipy.ndimage
from utils import SimCCLabel

# SimCC Head
class SimCC(nn.Module):
    def __init__(self, in_channels, input_size, in_feature_map_size, split_ratio):
        super(SimCC, self).__init__()

        self.in_featuremap_size = in_feature_map_size
        self.flatten_size = self.in_featuremap_size[0] * self.in_featuremap_size[1]
        self.split_ratio = split_ratio

        # Linear layers for x and y coordinate classification
        self.fc_x = nn.Linear(self.flatten_size, int(input_size[0] * split_ratio))
        self.fc_y = nn.Linear(self.flatten_size, int(input_size[1] * split_ratio))

    def forward(self, x):
        x = torch.flatten(x, start_dim=2)

        x_coords = self.fc_x(x)
        y_coords = self.fc_y(x)

        return x_coords, y_coords

# Pose estimation model with resnet50 backbone and simcc head
class PoseSimCC(nn.Module):
    def __init__(self, num_joints=17, input_size=(192, 256), sigma=6.0, split_ratio=2.0, deconv_layers_cfg=None):
        super(PoseSimCC, self).__init__()
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.in_channels = 2048

        self.deconv_layers = self._make_deconv_layers(deconv_layers_cfg)

        self.final_layer = nn.Conv2d(
            in_channels=deconv_layers_cfg['num_filters'][-1],
            out_channels=num_joints,
            kernel_size=1
        )

        in_feature_map_size = ((input_size[0] // 32) * (2 ** deconv_layers_cfg['num_layers']), (input_size[1] // 32) * (2 ** deconv_layers_cfg['num_layers']))
        self.head = SimCC(in_channels=self.final_layer.out_channels, input_size=input_size, in_feature_map_size=in_feature_map_size, split_ratio=split_ratio)

        self.decoder = SimCCLabel(input_size, sigma=sigma, split_ratio=split_ratio, decode_visibility=True)

    def _make_deconv_layers(self, cfg):
        if cfg is None:
            return None
 
        layers = []
        in_channels = self.in_channels
        
        for i in range(cfg['num_layers']):
            out_channels = cfg['num_filters'][i]
            kernel_size = cfg['kernel_sizes'][i]
            stride = 2
            padding = (kernel_size - stride) // 2
            output_padding = 0

            layers.append(
                nn.ConvTranspose2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=padding, 
                    output_padding=output_padding, 
                    bias=cfg['with_bias']
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        if self.deconv_layers is not None:
            x = self.deconv_layers(x)
        x = self.final_layer(x)
        x_coords, y_coords = self.head(x)

        return x_coords, y_coords

    def predict(self, input):
        simcc_x, simcc_y = self.forward(input)

        simcc_x = simcc_x.detach().cpu().numpy()
        simcc_y = simcc_y.detach().cpu().numpy()
        
        keypoints, eval = self.decoder.decode(simcc_x, simcc_y)
        
        return (simcc_x, simcc_y), keypoints, eval

# Used configs for deconvolution layers of trained simcc pose model
deconv_layers_cfg = {
    'num_layers': 2,
    'num_filters': [1024, 512],
    'kernel_sizes': [4, 4],
    'with_bias': False
}