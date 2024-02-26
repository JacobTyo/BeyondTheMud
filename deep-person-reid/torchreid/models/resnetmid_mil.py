from __future__ import division, absolute_import
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchreid.models.set_transformer import BagMultiheadAttention
from torchreid.models.resnetmid import ResNetMid, Bottleneck, init_pretrained_weights

__all__ = ['resnet50mid_mil']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNetMid_mil(ResNetMid):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
            self,
            num_classes,
            loss,
            block,
            layers,
            last_stride=2,
            fc_dims=None,
            batch_size=1,
            bag_size=1,
            num_heads=8,
            acc_fn='set_transformer',
            **kwargs
    ):
        super(ResNetMid_mil, self).__init__(num_classes,
                                        loss,
                                        block,
                                        layers,
                                        last_stride=last_stride,
                                        fc_dims=fc_dims,
                                        **kwargs)
        self.single_batch_attn_size = self.feature_dim
        if acc_fn == 'set_transformer':
            self.attention = BagMultiheadAttention(self.single_batch_attn_size, num_heads=num_heads)
        elif acc_fn == 'max':
            # Use Max Pooling for bag representation
            self.attention = lambda x: torch.max(x, dim=1)[0]
        elif acc_fn == 'avg':
            # Use the average bag representation
            self.attention = lambda x: torch.mean(x, dim=1)
        else:
            raise NotImplementedError(f'Accumulation function {acc_fn} not implemented.')

        self.bag_classifier = nn.Linear(self.single_batch_attn_size, num_classes)

    def bag_attention(self, bags_of_instances):
        bag_features = self.attention(bags_of_instances)
        # and also just get the average bag representation from the bags_of_instances
        # bag_features = bags_of_instances.mean(dim=1)
        bag_predictions = self.bag_classifier(bag_features)
        return bag_predictions, bag_features


def resnet50mid_mil(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNetMid_mil(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=[1024],
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model
