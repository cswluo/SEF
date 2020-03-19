import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
#from .utils import load_state_dict_from_url
import pdb
import torch.nn.functional as torchf
from utils.misc import SoftSigmoid


device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
eps = torch.finfo().eps
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

########################################################################################################################

class SparseLoss(nn.Module):
    ''' forcing the combination weights to have the sparsity property.
    '''
    def __init__(self, rho=0, nparts=1):
        super(SparseLoss, self).__init__()
        self.rho = rho
        self.nparts = nparts

    def forward(self, x):
        # x should be a n_parts * n_features  matrix

        # method I: hope each map only contributes to one part of the object
        # loss = (self.rho * torch.sum(torch.abs(torch.sum(torch.abs(x), dim=0) - 1))).to(device=device)

        # method II: 
        loss = (self.rho * torch.sum(torch.abs(x))).to(device=device)

        return loss

class DistLoss(nn.Module):

    def __init__(self, rho=0, nparts=1):
        super(DistLoss, self).__init__()
        self.rho = rho
        self.nparts = nparts

    def forward(self, x):
        # x: nparts * n_features
        # pdb.set_trace()
        # normalizing data along the feature dimension
        x_norm = torch.norm(x, dim=1)
        x_norm_matrix = torch.ger(x_norm, x_norm)

        x_inner = torch.mm(x, x.T)
        self_similarity = torch.abs(torch.div(x_inner, x_norm_matrix))
        n_pairs = (torch.numel(self_similarity) - self.nparts)/2
        loss = (self.rho * (torch.sum(self_similarity) - torch.diagonal(self_similarity).sum())/2/n_pairs).to(device=device)

        return loss

class AttentionFeatures(nn.Module):

    def __init__(self, in_channels, nparts=0):
        super(AttentionFeatures, self).__init__()
        self.nparts = nparts
        self.conv1 = nn.Conv2d(in_channels, self.nparts, kernel_size=1, stride=1, bias=True)
        # self.conv1 = conv1x1(in_channels, self.nparts)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        

    def forward(self, x):

        # pdb.set_trace()
        # x: nbatch * nchannel * height * width
        nbatch, nchannel, height, width = x.shape
        maps = self.conv1(x).view(nbatch, -1, self.nparts)  # nbatch *  (height * width) * nparts

        # normalization
        # maps = self.softmax(maps) # softmax normalization
        # maps = torch.div(maps, torch.norm(maps, dim=1, keepdim=True))   # l2 norm
        # maps = torch.div(maps, torch.max(maps, dim=1, keepdim=True)[0]+eps) # l1 norm
        maps = self.sigmoid(maps)
        # maps = SoftSigmoid(maps, self.weight, self.bias)
        
        x_view = x.view(nbatch, nchannel, -1) # nbatch, nchannel, height*width

        # attention pooling features
        x = torch.div(torch.bmm(x_view, maps).view(nbatch, self.nparts, nchannel), height*width)  # nbatch * nparts * nchannel

        # return attention features, combination weights, attention maps
        return x, torch.squeeze(self.conv1.weight), maps.view(nbatch, self.nparts, height, width)


#############################################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nparts=0, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attention=False):
        super(ResNet, self).__init__()

        self.attention = attention

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        if self.attention:
            self.nparts = nparts
            self.attention_cbns = AttentionFeatures(256*block.expansion, nparts=self.nparts)
            attention_predictor = []
            for i in range(self.nparts):
                attention_predictor.append(nn.Linear(256*block.expansion, num_classes))
            self.attention_fc = nn.Sequential(*attention_predictor)

            self.layer4_filter = conv1x1(512*block.expansion, 256*block.expansion)
            self.layer3_filter = conv1x1(256*block.expansion, 256*block.expansion)
            self.layer3_smoother = conv3x3(256*block.expansion, 256*block.expansion)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # pdb.set_trace()

        if self.attention:
            semantic3 = self.layer3_filter(x)

        x = self.layer4(x)
        
        if self.attention:

            semantic4 = torchf.interpolate(self.layer4_filter(x), 14, mode='bilinear')
            semantic34 = self.layer3_smoother(semantic3+semantic4)

            # x_clone = x.clone()
            attention_features, attention_weights, attention_maps = self.attention_cbns(semantic34)

            attention_scores = []
            for i in range(len(self.attention_fc)):
                attention_scores.append(self.attention_fc[i](torch.squeeze(attention_features[:,i,:])))
            
            
            # for original resnet outputs
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
            # method I:
            # # pdb.set_trace()
            attention_scores.append(x)
            # add predictive logits from every part and the object
            # x = torch.sum(torch.stack(attention_scores), dim=0)
            x = torch.roll(torch.stack(attention_scores, dim=0), 1, dims=0)

            # method II:
            # pdb.set_trace()
            # scores = list()
            # scores.append(x)
            # scores.extend(attention_scores)
            # x = torch.stack(scores)

            return x, attention_weights, attention_maps
        else:
            # for original resnet outputs
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x



def _resnet(arch, block, layers, pretrained, progress, model_dir=None, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[arch], model_dir=model_dir))
        # state_dict = load_state_dict_from_url(model_urls[arch],
        #                                       progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, model_dir=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, model_dir,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, model_dir=None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, model_dir,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
