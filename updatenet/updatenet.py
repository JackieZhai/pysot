
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn

class UpdateResNet(nn.Module):
    def __init__(self, config=None):
        super(UpdateResNet, self).__init__()
        self.update = nn.Sequential(
            nn.Conv2d(768, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 1),
        )
    def forward(self, x, x0):
        #t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.update(x)        
        response += x0
        return response
