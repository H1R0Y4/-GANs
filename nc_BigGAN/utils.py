from torch import nn


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data)
    elif classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)