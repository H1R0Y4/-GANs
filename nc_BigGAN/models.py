import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from layers import *


class Generator(nn.Module):
    def __init__(self, ngpu, ch=64, dim_z=128):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.sn_linear = spectral_norm(nn.Linear(dim_z, 4 * 4 * 16 * ch, bias=True))
        self.res0 = ResBlock_UP(16 * ch, 16 * ch)
        self.res1 = ResBlock_UP(16 * ch, 8 * ch)
        self.res2 = ResBlock_UP(8 * ch, 8 * ch)
        self.res3 = ResBlock_UP(8 * ch, 4 * ch)
        self.res4 = ResBlock_UP(4 * ch, 2 * ch)
        self.attn = SelfAttention(2 * ch)
        self.res5 = ResBlock_UP(2 * ch, ch)

        self.output = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=False),
            spectral_norm(
                nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1, bias=True)
            ),
            nn.Tanh(),
        )

    def forward(self, z, y):
        h = self.sn_linear(z)
        h = h.view(h.size(0), -1, 4, 4)
        h = self.res0(h, y)
        h = self.res1(h, y)
        h = self.res2(h, y)
        h = self.res3(h, y)
        h = self.res4(h, y)
        h = self.attn(h)
        h = self.res5(h, y)
        h = self.output(h)
        return h


class Discriminator(nn.Module):
    def __init__(self, ngpu, ch=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.res0 = ResBlock_DOWN(3, ch, preactivation=False, down=True)  # output torch.Size([b, 64, 128, 128])
        self.res1 = ResBlock_DOWN(ch, 2 * ch, preactivation=True, down=True)  # output torch.Size([b, 128, 64, 64])
        self.attn = SelfAttention(ch * 2)  # 論文通り 解像度64*64にattention層を追加
        self.res2 = ResBlock_DOWN(2 * ch, 4 * ch, preactivation=True, down=True)  # output torch.Size([b, 256, 32, 32])
        self.res3 = ResBlock_DOWN(4 * ch, 8 * ch, preactivation=True, down=True)  # output torch.Size([b, 512, 16, 16])
        self.res4 = ResBlock_DOWN(8 * ch, 8 * ch, preactivation=True, down=True)  # output torch.Size([b, 512, 8, 8])
        self.res5 = ResBlock_DOWN(8 * ch, 16 * ch, preactivation=True, down=True)  # output torch.Size([b, 1024, 4, 4])
        self.res6 = ResBlock_DOWN(16 * ch, 16 * ch, preactivation=True, down=False)  # output torch.Size([b, 1024, 4, 4])

        self.sn_linear = spectral_norm(nn.Linear(16 * ch, 1, bias=True))
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        # Stick x into h for cleaner for loops without flow control
        h = x
        self.res0(h)
        h = self.res0(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.attn(h)
        h = self.res5(h)
        h = self.res6(h)
        # global sum pool
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        h = self.sn_linear(h)
        return h