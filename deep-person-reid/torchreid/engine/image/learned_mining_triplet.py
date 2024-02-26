from __future__ import division, print_function, absolute_import

import os.path as osp

from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss

import torch
import torch.nn.functional as F

import learn2learn as l2l


from ..engine import Engine
from ...utils import save_checkpoint


class ImageTripletEngineLearnedMining(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        accumulator_fn=None,
        accumulator_optimizer=None,
        accumulator_lr=None,
        inner_steps=5,
        first_order_approx=True,
        inner_learning_rate=0.01,
        per_sample_loss_fn=None,
        val_risk_function=None
    ):
        super(ImageTripletEngineLearnedMining, self).__init__(datamanager, use_gpu)

        self.model = l2l.algorithms.MAML(model, lr=inner_learning_rate, first_order=first_order_approx)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.inner_step_counter = 0
        self.learner = None

        self.acc_fn = accumulator_fn
        self.acc_optim = accumulator_optimizer
        self.acc_lr = accumulator_lr
        self.inner_steps = inner_steps
        self.per_sample_loss_fn = per_sample_loss_fn
        self.val_risk_function = val_risk_function

        assert self.acc_fn is not None, 'A learnable accumulator function must be provided'
        assert self.acc_optim is not None, 'An optimizer for the accumulator function must be provided'
        assert self.per_sample_loss_fn is not None, 'A per sample loss function must be provided'
        assert self.val_risk_function is not None, 'A validation risk function must be provided'

    def forward_backward(self, data):

        # this needs to get a lot more complicated. Basically, it is going to do a 5 step look ahead.
        # gotta be a little careful, because we only get one batch of data as input, so gotta maintain an
        # internal counter to track if we are on an inner or an outer loop.

        imgs, pids = self.parse_data_for_train(data)

        loss_summary = {}

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        if not self.model.training:
            outputs, features = self.model(imgs)
            loss = self.per_sample_loss_fn(outputs, pids)
            loss = self.acc_fn(loss)

            loss_summary['learned_loss'] = loss.item()

        else:
            if self.inner_step_counter == 0:
                self.learner = self.model.clone()
                self.acc_optim.zero_grad()

            outputs, features = self.learner(imgs)

            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

            # outputs = F.log_softmax(outputs, dim=1)

            if self.inner_step_counter >= self.inner_steps:
                self.inner_step_counter = 0
                # we have finished the inner loop, so we need to do an outer step
                loss = self.val_risk_function(outputs, pids)

                loss_summary['val_loss'] = loss.item()

                loss.backward()
                self.acc_optim.step()
                self.acc_optim.zero_grad()
                self.optimizer.zero_grad()

                # and reuse the data to update the model
                outputs, features = self.model(imgs)
                loss = self.per_sample_loss_fn(outputs, pids)
                # instead of sort, we need to apply everything to an even simpler accumulator
                # the accumulator must operate only on a batch of data sorted by identity?
                # should we say "oh for these triplets, originating from the same class, you should do  x"?
                # no maybe not, that doesn't make sense for tripelts.
                loss, _ = torch.sort(loss, descending=True)
                loss = self.acc_fn(loss)
                loss_summary['learned_loss'] = loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                loss = self.per_sample_loss_fn(outputs, pids)

                # reorder the loss by magnitude for the accumulator
                loss, _ = torch.sort(loss, descending=True)

                # accumulate the loss via our learnable accumulator
                loss = self.acc_fn(loss)

                loss_summary['learned_loss'] = loss.item()

                self.learner.adapt(loss)

            self.inner_step_counter += 1

        return loss_summary

    def save_model(self, epoch, rank1, save_dir, is_best=False):
        # just need to also make sure to save the accumualtors
        super(ImageTripletEngineLearnedMining, self).save_model(epoch, rank1, save_dir, is_best)

        save_checkpoint(
            {
                'state_dict': self.acc_fn.state_dict(),
                'epoch': epoch + 1,
                'rank1': rank1,
                'optimizer': self.acc_optim.state_dict()
            },
            osp.join(save_dir, 'accumulator'),
            is_best=is_best
        )

    def set_model_mode(self, mode='train', names=None):
        # use normal but also set this for the accumulation function
        super(ImageTripletEngineLearnedMining, self).set_model_mode(mode, names)
        if mode == 'train':
            self.acc_fn.train()
        else:
            self.acc_fn.eval()

