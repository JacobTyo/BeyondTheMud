from __future__ import division, absolute_import
import torch
from torch import nn
from torchreid.models.osnet import OSNet, OSBlock, init_pretrained_weights
from torchreid.models.set_transformer import BagMultiheadAttention


class OSNetMil(OSNet):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        num_heads=8,
        batch_size=1,
        bag_size=1,
        acc_fn='set_transformer',
        **kwargs
    ):
        super(OSNetMil, self).__init__(num_classes, blocks, layers, channels, **kwargs)
        self.single_batch_attn_size = self.feature_dim
        if acc_fn == 'set_transformer':
            self.attention = BagMultiheadAttention(self.single_batch_attn_size, num_heads=num_heads)
        elif acc_fn == 'max':
            # Use Max Pooling for bag representation
            self.attention = lambda x: torch.max(x, dim=1)[0]
        elif acc_fn == 'avg':
            # Use the average bag representation
            self.attention = lambda x: torch.mean(x, dim=1)
        elif acc_fn == 'sum':
            # Use the sum of the bag representation
            self.attention = lambda x: torch.sum(x, dim=1)
        else:
            raise NotImplementedError(f'Accumulation function {acc_fn} not implemented.')

        self.bag_classifier = nn.Linear(self.single_batch_attn_size, num_classes)

    def bag_attention(self, bags_of_instances):
        bag_features = self.attention(bags_of_instances)
        # and also just get the average bag representation from the bags_of_instances
        # bag_features = bags_of_instances.mean(dim=1)
        bag_predictions = self.bag_classifier(bag_features)
        return bag_predictions, bag_features


def osnet_x1_0_mil(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNetMil(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model
