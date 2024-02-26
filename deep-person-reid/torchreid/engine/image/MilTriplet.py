from __future__ import division, print_function, absolute_import

import torch
from torch.autograd import Variable
from torchreid import metrics
from torchreid.losses import TripletLoss, CrossEntropyLoss, BagInstanceDifferenceLoss

import numpy as np
from tqdm import tqdm

from ..engine import Engine


class ImageMilTripletEngine(Engine):
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
            weight_d=1,
            scheduler=None,
            use_gpu=True,
            label_smooth=True,
            bag_size=None
    ):
        super(ImageMilTripletEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.bag_size = bag_size
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0 and weight_d >= 0
        assert weight_t + weight_x + weight_d > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_d = weight_d

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_d = BagInstanceDifferenceLoss(use_gpu=self.use_gpu)

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        # for every instance, get the representation
        outputs, features = self.model(imgs)

        # Group features by pids (bag IDs)
        bag_features_dict = {}
        bag_pids = []
        for feature, pid in zip(features, pids):
            pid_key = pid.item()
            if pid_key not in bag_features_dict:
                bag_features_dict[pid_key] = []
                bag_pids.append(pid_key)
            # so this is a list of tensors, each of shape [feature_dim]
            bag_features_dict[pid_key].append(feature)

        # now make sure each bag has the same number of instances = self.bag_size
        if self.bag_size is not None:
            for pid in bag_features_dict:
                # cast the list of tensors to a list of tensors each of shape [number_of_instances, feature_dim]
                # so shape is [num_bags_this_pid, number_of_instances, feature_dim]
                bag_features_dict[pid] = [torch.vstack(bag_features_dict[pid][x:x+self.bag_size])
                                          for x in range(0, len(bag_features_dict[pid]), self.bag_size)]

        # now get the bag_pid list and bag_features lists for input to attention
        grouped_bag_features = []
        grouped_bag_labels = []
        for pid, feature_list in bag_features_dict.items():
            for feat in feature_list:
                grouped_bag_features.append(feat)
                grouped_bag_labels.append(pid)

        # and cast to tensor
        grouped_bag_features = torch.stack(grouped_bag_features, dim=0)
        grouped_bag_labels = torch.tensor(grouped_bag_labels)

        # now, we need to pass each bag through the attention network to get the bag representation
        # the input should be of shape [number_of_bags, number_of_instances, feature_dim]
        bag_predictions, bag_features = self.model.module.bag_attention(grouped_bag_features)

        # now, we need to compute the loss. We can give the pid of each bag and the bag features to the loss function
        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, bag_features, grouped_bag_labels)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        # if self.weight_x > 0:
        loss_x = self.compute_loss(self.criterion_x, bag_predictions, grouped_bag_labels)
        loss += self.weight_x * loss_x
        loss_summary['loss_x'] = loss_x.item()
        loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        loss_d = self.criterion_d(bag_features, grouped_bag_features)
        if self.weight_d > 0:
            loss += self.weight_d * loss_d
        loss_summary['loss_d'] = loss_d.item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary

    def test(self, *args, **kwargs):
        super(ImageMilTripletEngine, self).test(*args, **kwargs)
        # if 'sysu30k' in self.datamanager.sources:
        #     # follow the sysu30k evaluation procedure, otherwise, normal evaluation
        #     self.test_sysu30k(*args, **kwargs)
        # else:
        #     super(ImageMilTripletEngine, self).test(*args, **kwargs)

    @torch.no_grad()
    def test_sysu30k(self, *args, ranks=[1, 5, 10, 20], **kwargs):

        # Some helper functions just for this eval
        def sysu_extract_feature(model, dataloaders):
            for i, data in enumerate(tqdm(dataloaders)):
                img, label, cam_id = data['img'], data['pid'], data['camid']
                outputs = model(img)
                if i == 0:
                    features = torch.zeros((len(dataloaders) * outputs.size(0), outputs.size(1)), dtype=torch.float32)
                features[i * outputs.size(0): (i + 1) * outputs.size(0), :] = outputs.data.cpu()
            return features

        # Evaluate
        def sysu_evaluate(qf, ql, qc, gf, gl, gc):
            query = qf
            score = np.dot(gf, query)
            # predict index
            index = np.argsort(score)  # from small to large
            index = index[::-1]
            # index = index[0:2000]
            # good index
            query_index = np.argwhere(gl == ql)
            camera_index = np.argwhere(gc == qc)

            # this removes any images of the same label, but also of the same camera index
            # so we want only the best match on an image from a different camera
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            junk_index1 = np.argwhere(gl == -1)
            junk_index2 = np.intersect1d(query_index, camera_index)
            junk_index = np.append(junk_index2, junk_index1)  # .flatten())

            _ap, _cmc = sysu_compute_mAP(index, good_index, junk_index)
            return _ap, _cmc

        def sysu_compute_mAP(index, good_index, junk_index):
            ap = 0
            cmc = torch.IntTensor(len(index)).zero_()
            if good_index.size == 0:  # if empty
                cmc[0] = -1
                return ap, cmc

            # remove junk_index
            mask = np.in1d(index, junk_index, invert=True)
            index = index[mask]

            # find good_index index
            ngood = len(good_index)
            mask = np.in1d(index, good_index)
            rows_good = np.argwhere(mask == True)
            rows_good = rows_good.flatten()

            for i in range(ngood):
                d_recall = 1.0 / ngood
                precision = (i + 1) * 1.0 / (rows_good[i] + 1)
                if rows_good[i] != 0:
                    old_precision = i * 1.0 / rows_good[i]
                else:
                    old_precision = 1.0
                ap = ap + d_recall * (old_precision + precision) / 2

            return ap, cmc

        self.model.eval()

        # Extract feature
        test_dataset = self.test_loader.keys()
        assert len(test_dataset) == 1, 'Only support one test dataset'
        test_dataset = list(test_dataset)[0]
        print('extracting gallery features...')
        gallery_feature = sysu_extract_feature(self.model, self.test_loader[test_dataset]['gallery'])
        print('extracting query features...')
        query_feature = sysu_extract_feature(self.model, self.test_loader[test_dataset]['query'])

        query_feature = query_feature.numpy()  # result['query_f']
        gallery_feature = gallery_feature.numpy()

        print('gathering query labels and camera info')
        query_label, query_cam = [], []
        for data in self.test_loader[test_dataset]['query']:
            query_label.extend(data['pid'].numpy().tolist())
            query_cam.extend(data['camid'].numpy().tolist())

        print('gathering gallery labels and camera info')
        gallery_label, gallery_cam = [], []
        for data in tqdm(self.test_loader[test_dataset]['gallery']):
            gallery_label.extend(data['pid'].numpy().tolist())
            gallery_cam.extend(data['camid'].numpy().tolist())

        # Trim the extra "features" if the last batch was not full.
        gallery_feature = gallery_feature[:len(query_label)]

        print('continuing')
        query_cam = np.asarray(query_cam)
        query_label = np.asarray(query_label)
        gallery_cam = np.asarray(gallery_cam)
        gallery_label = np.asarray(gallery_label)

        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        # print(query_label)
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = sysu_evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label,
                                       gallery_cam)
            if CMC_tmp[0] == -1:
                continue

            cmc_size = CMC.size(0)
            cmc_tmp_size = CMC_tmp.size(0)
            if cmc_tmp_size < cmc_size:
                padding = cmc_size - cmc_tmp_size
                CMC_tmp = torch.cat((CMC_tmp, torch.zeros(padding, dtype=CMC_tmp.dtype)), 0)

            CMC = CMC + CMC_tmp
            ap += ap_tmp
            print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC / len(query_label)  # average CMC

        print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))

        if self.writer is not None:
            for r in ranks:
                self.writer.add_scalar(f'Test/sysu30k/rank{r}', CMC[r - 1], self.epoch)
            self.writer.add_scalar(f'Test/sysu30k/mAP', ap / len(query_label), self.epoch)


