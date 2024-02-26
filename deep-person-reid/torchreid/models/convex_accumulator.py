import torch
import torch.nn as nn


class ConvexAccumulator(nn.Module):

    def __init__(self, batch_size, normalization='softmax', update_frequency=1):
        super(ConvexAccumulator, self).__init__()
        self.batch_size = batch_size
        self.layer = nn.Linear(batch_size, 1)
        self.normalization = normalization
        self.counter = 0
        self.update_freq = update_frequency
        # make convex combination
        self.normalize_weights()

    def normalize_weights(self, force=False):
        # ensure this operation equates to a convex combination
        if self.normalization == 'softmax':
            return
        if ((self.counter % self.update_freq) == 0) or force:
            with torch.no_grad():
                if self.normalization == 'threshold':
                    tmp = self.layer.weight.detach()
                    tmp = tmp - torch.min(tmp)
                    tmp = tmp / tmp.sum()
                    self.layer.weight.data = tmp
                elif self.normalization == 'relu':
                    tmp = self.layer.weight.detach()
                    tmp = nn.functional.relu(tmp)
                    tmp = tmp / tmp.sum()
                    self.layer.weight.data = tmp
                else:
                    raise ValueError('the supported normalization methods are "reweight" and "softmax"')

    def get_normalized_weights(self):
        self.normalize_weights(True)
        weights = self.layer.weight.detach().cpu()
        if self.normalization == 'softmax':
            weights = nn.functional.softmax(weights, dim=1)
        return weights

    def forward(self, x):
        if self.training:
            self.normalize_weights()
            self.counter += 1

        if self.normalization == 'softmax':
            x = nn.functional.softmax(self.layer.weight, dim=1) @ x
        else:
            x = self.layer(x)
        return x
