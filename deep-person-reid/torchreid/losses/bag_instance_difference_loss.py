from __future__ import division, absolute_import
import torch
import torch.nn as nn


class BagInstanceDifferenceLoss(nn.Module):
    r"""Given a bag representatin, and then the representation of all images in a bag,
    use the distance between the bag representation and the closest image representation
    as the loss.

    Args:
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
    """

    def __init__(self, use_gpu=True):
        super(BagInstanceDifferenceLoss, self).__init__()
        self.use_gpu = use_gpu
        self.euc_dist = torch.cdist

    def forward(self, bags, bag_instances):
        """
        Args:
            bags (torch.Tensor): matrix with shape (batch_size, bag_feature_size)
            bag_instances (torch.LongTensor): matrix with the shape (batch_size, num_instances, instance_feature_size)
        """
        bags = bags.unsqueeze(1)
        dist = self.euc_dist(bags, bag_instances)
        dist = dist.min(dim=2)[0]

        return dist.mean()
