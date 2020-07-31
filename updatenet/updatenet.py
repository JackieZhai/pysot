
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn

class UpdateResNet(nn.Module):
    def __init__(self, config=None):
        super(UpdateResNet, self).__init__()
        self.update1 = nn.Sequential(
            nn.Conv2d(768, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 256, 1),
        )
        # self.update2 = nn.Sequential(
        #     nn.Conv2d(512, 96, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(96, 256, 1),
        # )
        # self.update3 = nn.Sequential(
        #     nn.Conv2d(512, 96, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(96, 256, 1),
        # )
    def forward(self, x, x0):
        #t = torch.cat((x, y, z), 0)
        # x0 is residual
        response = self.update1(x)
        # x1 = x[:,256:512,:,:]
        # x2 = x[:,512:768,:,:]
        # response = torch.cat([response, x1], dim=1)
        # response = self.update2(response)
        # response = torch.cat([response, x2], dim=1)
        # response = self.update3(response)
        response += x0
        return response
