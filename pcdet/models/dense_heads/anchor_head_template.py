import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        # 网络类别预测
        # Network Category Forecast
        cls_preds = self.forward_ret_dict['cls_preds']
        # 前景anchor的类别
        # categories of foreground anchor
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        # 我们所关心的anchor
        # 选出前景背景anchor，在0.45到0.6之间时仍置为-1，并不参与loss计算
        # the anchors we concerned: Get the anchor of foreground and background,
        # and set -1 for anchors with IoU values between 0.45 and 0.6,
        # which are not involved in the loss calculation.
        cared = box_cls_labels >= 0  # [N, num_anchors]
        # 前景anchor
        # The anchor of the foreground
        positives = box_cls_labels > 0
        # 背景anchor
        # The anchor of the background
        negatives = box_cls_labels == 0
        # 背景anchor赋予权重
        # Assign weights to the anchor of the background.
        negative_cls_weights = negatives * 1.0
        # 将每个anchor分类的损失权重都置为1
        # set the loss weight of each anchor's classification to 1
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # 每个正样本anchor的回归损失权重置为1
        # set the loss weight of each positive sample anchor regression to 1
        reg_weights = positives.float()
        # 如果只有一类
        # When the classification result has only one category
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        # 正则化并计算权重
        # 求出每个数据中有多少个正例
        # The regularization operation is performed and the weight value is calculated,
        # and then the number of positive examples for each data center is calculated.
        pos_normalizer = positives.sum(1, keepdim=True).float()
        # 正则化回归损失，最小值为1，用正样本数量来正则化回归损失
        # regularize the regression loss with the number of positive samples
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # 正则化分类损失，最小值为1，用正样本数量来正则化分类损失
        # regularize the classification loss with the number of positive samples
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # care包含了背景和前景的anchor，但是这里只需要获取前景部分的类别即可，不必关注-1还是0
        # cared.type_as(box_cls_labels)，cared中为False的部分因不需要计算loss的anchor，故置为0
        # 对应位置相乘后，所有背景和IoU介于match_threshold和unmatch_threshold之间的anchor均置为0
        # cared contains both foreground and background anchors,
        # and we only need to focus on the category of the foreground part of it,
        # and we do not care whether it is -1 or 0.
        # cared.type_as(box_cls_labels): for the False part of cared, the anchors
        # that are not involved in the loss calculation should all be set to 0.
        # After multiplying the parameters in the corresponding positions,
        # all the anchors between match_threshold and unmatch_threshold are set to 0 for all backgrounds and IoUs.
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        # 在最后一个维度上扩展一次
        # Perform an extended operation on the last dimension
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        # +1是考虑到背景
        # self.num_class + 1: take into account the context.
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )   
    
        # scatter_函数的一个典型应用是用于分类问题
        # 将目标标签转换为one-hot编码形式，这里表示在最后一个维度，将cls_targets.unsqueeze(dim=-1)所索引的位置置为1
        # A typical application of the scatter_ function is the classification problem,
        # where the labels of the targets are converted to the one-hot encoded form.
        # This part is mainly used in the last dimension,
        # and then the positions of the cls_targets.unsqueeze(dim=-1) indexes are all set to 1.
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0) 
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        # 不计算背景分类损失
        # This procedure does not involve the calculation of background classification loss.
        one_hot_targets = one_hot_targets[..., 1:]

        # 计算分类损失
        # calculate classification losses
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        # 求和并除以batch数目
        # sum up and divide by the number of batches
        cls_loss = cls_loss_src.sum() / batch_size
        # loss乘以分类权重
        # loss multiplied by the classification weight
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        # 考虑到reg_targets[..., 6]是经过编码的旋转角度，如果要回到原始角度需要重新加回anchor的角度即可
        # Considering that reg_targets[... , 6] is the rotation angle obtained after encoding,
        # if you need to convert to the original angle, you need to add the anchor angle back in.
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        # num_bins = 2
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            # one-hot编码，只存在两个方向：正向和反向
            # For one-hot encoding, there are only two directions, forward and reverse.
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        # anchor_box的7个回归参数
        # 7 arguments of anchor_box
        box_preds = self.forward_ret_dict['box_preds']
        # anchor_box的方向预测
        # orientation prediction of anchor_box
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        # 每个anchor和GT编码的结果
        # the coding results for each anchor and GT
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        # 获取所有anchor中属于前景anchor的mask
        # get the mask of the foreground anchor in all anchors
        positives = box_cls_labels > 0
        # 设置回归参数为1
        # set the regression parameter to 1
        reg_weights = positives.float()                             # keep only the values with labels greater than 0
        pos_normalizer = positives.sum(1, keepdim=True).float()     # calculate the sum of all positive samples
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        
        box_loss = loc_loss
        tb_dict = {
            # PyTorch中的item()方法，返回张量中的元素值，与python中针对dict的item方法不同
            # Use the item() method in PyTorch to return the values of the elements in the tensor.
            'rpn_loss_loc': loc_loss.item()
        }

        # 如果存在方向预测，则添加方向损失
        # If a directional forecast exists, a directional loss should be added.
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,       # Directional Offset
                num_bins=self.model_cfg.NUM_DIR_BINS        # the number of the direction of BINS
            )
            # 方向预测值
            # the predicted value of the direction
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            # 只要正样本的方向预测值
            # only get the directional prediction of positive samples
            weights = positives.type_as(dir_logits)
            # 除了正例的数量，使得每个样本的损失与样本中目标的数量无关
            # To make the loss of each sample independent of the number of targets in the sample,
            # the parameter is divided by the number of positive samples.
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # 方向损失计算
            # the calculation of the loss of direction
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            # 损失权重
            # the calculation of the weight of the loss
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']

            # 将方向损失加入box损失
            # add the loss of direction to the loss of box
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            # 判断是否需要使用多头预测，默认值为False。
            # Determines if multiple predictions need to be used, the default value is False.
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors

        # 计算anchor的总数量
        # calculate the total number of anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # 将预测结果展开为一维的数据张量
        # expand the prediction results into a one-dimensional data tensor
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        # 解码7个用于预测box的参数
        # decode the 7 parameters used to predict the box
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        # 对每个anchor的方向的预测
        # prediction of the direction of each anchor
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET                  # 方向偏移量，0.78539     # direction offset, 0.78539
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET      # 0
            # 将方向的预测结果展开为一维的张量
            # expand the prediction result of the direction into a one-dimensional tensor
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            # 获取所有anchor的方向分类结果，即正向和反向。
            # Get the directional classification results of all anchors, i.e. forward and reverse.
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)      # pi
            # 将角度规范在0到pi之间。在OpenPCDet中，坐标方向确定为统一的规范的坐标方向，即x向前，y向左，z向上。
            # Normalize the angle between 0 and pi. In OpenPCDet,
            # the coordinate direction is determined as a uniform canonical coordinate direction,
            # i.e. x forward, y left, z up.
            # 参考训练时的原理，将角度沿着x轴的逆时针方向旋转45度，进而得到dir_rot。
            # Referring to the principle during training,
            # the angle is rotated 45 degrees counterclockwise along the x-axis,
            # which in turn gives dir_rot.
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
