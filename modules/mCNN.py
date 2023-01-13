import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.block import conv_block, fc_block


class mCNN(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.hl_c1 = conv_block(1, 32, 7, bias=False)
        self.ll_c1 = conv_block(1, 32, 7, bias=False)
        self.lh_c1 = conv_block(1, 32, 7, bias=False)
        self.hh_c1 = conv_block(1, 32, 7, bias=False)

        self.c2 = conv_block(32, 16, 3, bias=False, pool_size=4, dropout=0.25)
        self.c3 = conv_block(16, 16, 3, bias=False)
        self.c4 = conv_block(16, 16, 3, bias=False, dropout=0.25)

        self.fc1 = fc_block(3072, 32, dropout=0.5)
        self.fc2 = fc_block(32, num_classes, activation=nn.Softmax(dim=1))

    def forward(self, x):
        ll, lh, hl, hh = x
        hl = self.hl_c1(hl)
        ll = self.ll_c1(ll)
        lh = self.lh_c1(lh)
        hh = self.hh_c1(hh)

        Max_LH_HL_HH = torch.max(lh, torch.max(hl, hh))
        x_merged = ll * Max_LH_HL_HH
        x = self.c2(x_merged)
        x = self.c3(x)
        x = self.c4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
