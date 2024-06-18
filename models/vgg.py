import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F

# __all__ = ['vgg19', 'vgg19_d', 'vgg19_0', 'vgg19_1']
__all__ = ['vgg19', 'vgg19_d', 'vgg19_r']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        # self.reg_layer_1 = SublinearSequential(
        #     *list(self.reg_layer_0.children())
        # )

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer_0(x)
        return torch.abs(x)


class VGG_R(nn.Module):
    def __init__(self, features):
        super(VGG_R, self).__init__()
        self.features = features
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer_0(x)
        return torch.relu(x)



class VGG_D(nn.Module):
    def __init__(self, features):
        super(VGG_D, self).__init__()
        self.features = features

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self.reg_layer_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_0 = self.features(x)
        x = self.reg_layer(x_0)
        var = self.reg_layer_1(x_0) * 16
        return torch.relu(x), var


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
        elif v == 'D':
            conv2d = nn.Conv2d(in_channels, 512, kernel_size=3, padding=2, dilation=2)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = 512
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 'D', 'D', 'D', 'D']
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


def vgg19_r():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_R(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

# def vgg19_0():
#     """VGG 19-layer model (configuration "E")
#         model pre-trained on ImageNet
#     """
#     model = VGG_0(make_layers(cfg['E']))
#     model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
#     return model
#
# def vgg19_1():
#     """VGG 19-layer model (configuration "E")
#         model pre-trained on ImageNet
#     """
#     model = VGG_1(make_layers(cfg['E']))
#     model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
#     return model



def vgg19_d():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG_D(make_layers(cfg['F']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
