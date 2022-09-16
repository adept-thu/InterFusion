import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        # anchor生成配置参数
        # anchor generates configuration parameters
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        # 为预测box找对应anchor的参数
        # find the parameters of the corresponding anchor to predict the box
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        # 编码box的7个残差参数(x,y,z,w,l,h,θ)
        # code the 7 residual parameters of the box
        self.box_coder = box_coder
        # 在PointPillars中指定正负样本的时候由BEV视角计算GT和先验框的IoU，不需要进行z轴上高度的匹配
        # When specifying positive and negative samples,
        # it is considered that the IoU of GT and a priori frame is calculated based on the BEV perspective,
        # so it is not necessary to consider the matching of the height in the z-axis direction.
        self.match_height = match_height
        # 获取类别名称['Car', 'Pedestrian', 'Cyclist']
        # get the type names['Car','Pedestrian','Cyclist']
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        # 设置前景、背景的采样系数（PointPillars不需要考虑该参数）
        # set the sampling coefficients for foreground and background
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        # 总采样数（PointPillars不需要考虑该参数）
        # set the total number of samples
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        # False：前景权重用1/前景anchor数量（PointPillars不需要考虑该参数）
        # set the weight of the foreground
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        # 设定类别的IoU匹配为正样本时的阈值{'Car': 0.6, 'Pedestrian':0.5, 'Cyclist':0.5}
        # Set the threshold when the IoU match of the category is a positive sample.
        # {'Car:0.6','Pedestrian:0.5','Cyclelist:0.5'}
        self.matched_thresholds = {}
        # 设定类别的IoU匹配为负样本时的阈值{'Car':0.45, 'Pedestrian':0.35. 'Cyclist':0.35}
        # Set the threshold when the IoU match of the category is a negative sample.
        # {'Car:0.45','Pedestrian:0.35','Cyclelist:0.35'}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
         
        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        self.seperate_multihead = model_cfg.get('SEPERATE_MULTIHEAD', False)
        if self.seperate_multihead:
            rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
            self.gt_remapping = {}
            for rpn_head_cfg in rpn_head_cfgs:
                for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
                    self.gt_remapping[name] = idx + 1

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        # 初始化结果list并提取对应的gt_box和类别
        # Initialize the result list and extract the corresponding gt_box and category.
        bbox_targets = []
        cls_labels = []
        reg_weights = []

        # 获取批次大小
        # get batch size
        batch_size = gt_boxes_with_classes.shape[0]
        # 获取所有GT的类别
        # get all categories of GT
        gt_classes = gt_boxes_with_classes[:, :, -1]
        #获取所有GT的7个box参数
        # get all 7 box parameters of GT
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        # 对batch中所有数据进行逐帧匹配anchor的前景和背景
        # Match the foreground and background of the anchor frame by frame for all data in the batch.
        for k in range(batch_size):
            # 取出当前帧中的gt_boxes
            # get the gt_boxes of the current frame
            cur_gt = gt_boxes[k]
            #获取一批数据中最多有多少个GT
            # get the maximum number of data center GTs in the same batch
            cnt = cur_gt.__len__() - 1
            # 通过循环操作来获取最后一个非零的box，因为预处理的时候会按照batch最大box的数量处理，当box数量不足会补充0来补足数量
            # Find the last non-zero box,
            # if the number of non-zero boxes is insufficient,
            # the missing box will be automatically filled with 0.
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            # 提取当前帧中非零的box和类别
            # extract the non-zero box and category of the current frame
            cur_gt = cur_gt[:cnt + 1]
            # 转换数据类型
            # convert data types
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            # 对每帧中的anchor和GT分类别，单独计算前景、背景
            # 因为每个类别的anchor是独立计算的，不同于在ssd中整体计算IoU并取最大值
            # The anchor and GT are classified for each frame,
            # while the front background is calculated separately.
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                # anchor_class_name: 车 | 行人 | 自行车骑行者
                # anchor_class_name: Car | Pedestrian | Cyclist
                if cur_gt_classes.shape[0] > 1:
                    # 减1是因为列表索引从0开始，将获取的属于列表中GT中与当前处理的类别相同的类别，从而得到类别mask
                    # self.class_names : ['Car','Pedestrian','Cyclist']
                    # In order to get the same minecrumbs belonging to the GT in the list as the current processed category,
                    # the list index is subtracted by 1 before processing, and finally the category mask is obtained.
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                # 判断检测头中是否需要使用多头，是的话设置为True；否则默认为False
                # determine whether to use multiple detection heads
                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    if self.seperate_multihead:
                        selected_classes = cur_gt_classes[mask].clone()
                        if len(selected_classes) > 0:
                            new_cls_id = self.gt_remapping[anchor_class_name]
                            selected_classes[:] = new_cls_id
                    else:
                        selected_classes = cur_gt_classes[mask]
                else:
                    # 计算所需变量，获取特征图大小
                    # calculate the required variables and get the feature map size
                    feature_map_size = anchors.shape[:3]
                    # 将所有的anchor展开
                    # flatten all the anchors
                    anchors = anchors.view(-1, anchors.shape[-1])
                    # List：根据类别的mask索引，获取该帧中档期啊需要处理的类别 --> 车 | 行人 | 自行车骑行者
                    # Get the current category to be processed in the frame based on the categorymask index.
                    # Car | Pedestrian | Cyclist
                    selected_classes = cur_gt_classes[mask]

                # 使用assign_targets_single单独为某一类别的anchor分配gt_boxes
                # 为前景、背景的box设置编码和回归权重
                # Use assign_targets_single to assign gt_boxes independently for a category of anchor.
                # Set coding and regression weights for foreground and background boxes.
                single_target = self.assign_targets_single(
                    anchors,                                                            # all the anchors of the category
                    cur_gt[mask],                                                       # GT_box
                    gt_classes=selected_classes,                                        # the currently selected category
                    # 当前类别anchor与GT匹配为正样本时的阈值
                    # the threshold of the current category of anchor matching with GT as a positive sample
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    # 当前类别anchor与GT匹配为负样本时的阈值
                    # the threshold of the current category of anchor matching with GT as a negative sample
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)
                # 到目前为止，处理完该帧单个类别和该类别anchor的前景和背景分配
                # Up to this position, we have finished assigning a single category of a given frame
                # with the foreground and background of the anchor of that category.

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)

            # 将结果填入对应的容器
            # Add the obtained results to the corresponding dictionaries in sequence.
            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
            #到这里，该batch的点云已经全部处理完毕
            # Up to this position, all point clouds for a given batch have been processed.

        # 将结果进行堆叠并返回
        # stack and return the results
        bbox_targets = torch.stack(bbox_targets, dim=0)

        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights

        }
        return all_targets_dict

    def assign_targets_single(self,
                         anchors,
                         gt_boxes,
                         gt_classes,
                         matched_threshold=0.6,
                         unmatched_threshold=0.45
                        ):

        # initialization
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]      # Number of GTs of the category in a frame

        # 初始化anchor对应的label和gt_id，并将其置为-1，表示loss计算时不会被考虑，且背景类别被置为0
        # Initialize the anchor's corresponding label and gt_id and set them to -1. 
        # The loss will not be considered in the calculation and the category of the background will be set to 0.
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        # 计算该类别中的anchor的前景和背景
        # calculate the anchor in the category
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 计算该帧找那个某一类别的GT和对应anchor之间的IoU（jaccard index）
            # calculate the IoU between the GT of a category and the corresponding anchor in the processing frame
            # anchor_by_gt_overlap: all anchors in the current category, and IoU of all GTs in the current category
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            
            # The speed of these two versions depends the environment and the number of anchors.
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()

            # 获取每一个anchor与哪个的GT的IoU的值最大
            # anchor_to_gt_argmax表示数据维度是anchor的长度，索引是GT
            # Get each anchor, and the maximum IoU of the GT corresponding to that anchor.
            # anchor_to_gt_argmax: the dimension of the data is equal to the length of the obtained anchor and the index is GT.
            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            # anchor_to_gt_max：获取每一个anchor最匹配的GT的IoU数值
            # anchor_to_gt_max: get the IoU value of the GT that best matches each anchor
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ]

            # 获取每个GT最匹配anchor的索引和IoU
            # 获取每个GT最匹配的anchor的索引
            # Get the index and IoU value of the most matching anchor for each GT.
            # get the index of the most matching anchor for each GT
            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            # 获取每个GT最匹配anchor的IoU
            # get the IoU value of the most matching anchor for each GT
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            # 将GT中没有匹配到的anchor的IoU置为-1
            # set -1 to the IoU value of the anchor that does not match in GT

            # 获取没有匹配到anchor的GT的mask
            # the mask of the GT of the anchor that did not match
            empty_gt_mask = gt_to_anchor_max == 0
            # 将没有匹配到anchor的GT的IoU置为-1
            # Set the IoU value of the GT of the anchor that is not matched to -1
            gt_to_anchor_max[empty_gt_mask] = -1

            # 获取anchor中和GT存在的最大IoU的anchor的索引，即前景anchor
            # nonzero函数为numpy中用于得到数组array中非零元素的位置（数组索引）的函数
            # Get the index of the anchor in the anchor that has the maximum IoU with GT, i.e., the foreground anchor.
            # nonzero: get the position of the non-zero element in the arrayarray (array index)
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            # 获取这些最匹配anchor与该类别对应的GT索引
            # get the most matching anchor and the corresponding GT index in that category
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            # 将GT索引也赋值到对应anchor的label中
            # add the category of GT to the label of the corresponding anchor
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            # 将GT索引也赋值到对应anchor的gt_ids中
            # add the index of GT to the gt_ids of the corresponding anchor
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            # 根据match_threshold和unmatch_threshold以及anchor_to_gt_max计算前景和背景索引，并更新label和gt_ids
            # calculate indexes of foreground and background, update label and gt_ids

            # 获取最匹配的anchor中IoU大于给定阈值的mask索引
            # Get the mask with IoU greater than the given threshold in the most matching anchor.
            pos_inds = anchor_to_gt_max >= matched_threshold
            # 获取最匹配的anchor中IoU大于给定阈值的GT索引
            # Get the index of the GT with IoU greater than the given threshold in the most matching anchor.
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            # 将pos anchor对应的GT索引赋值到对应的anchor的label中
            # Add the category of the GT corresponding to the pos_anchor to the label of the corresponding anchor.
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            # 将pos anchor对应的GT所以赋值到对应的anchor的gt_ids中
            # Add the index of the GT corresponding to the pos_anchor to the gt_id of the corresponding anchor.
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            # 获取背景anchor索引
            # get the index of the background anchor
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        # 获取前景anchor索引 --> (num_of_foreground_anchor)
        # 获取IoU介于unmatch_threshold和match_threshold之间的anchor，并进行下一步处理
        # get the index of the foreground anchor
        # get the IoU value between unmatch_threshold and match_threshold of the foreground anchor
        fg_inds = (labels > 0).nonzero()[:, 0]
        # 到这里，获取了属于前景或者背景的anchor
        # Up to this position, it has been confirmed whether part of the anchor belongs to the foreground or the background.

        # 对anchor的前景和背景进行筛选和赋值
        # 如果存在前景采样比例，则分别采样前景和背景anchor，PointPillars中没有前背景采样操作，前背景均衡使用了focal loss损失函数
        # filter and assign values to the foreground and background of the anchor
        # If there is a foreground sampling ratio,
        # the foreground and background anchors need to be sampled separately.
        # This operation is not available in PointPillars,
        # because the focal_loss function is used for foreground and background.
        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            # 如果前景anchor大于采样前景数
            # if the number of anchors in the foreground is greater than the number of sampled foregrounds,
            if len(fg_inds) > num_fg:
                # 计算要丢弃的前景anchor数目
                # calculate the number of foreground anchors to be dropped
                num_disabled = len(fg_inds) - num_fg
                # 在前景数目中随机产生索引值，取前num_disabled个并关闭索引
                # Generate random index values among the number of foregrounds,
                # then get the first num_disabled indexes.
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                # 将被丢弃的anchor的IoU置为-1
                # set -1 to the IoU of the anchor that was dropped
                labels[disable_inds] = -1
                # 更新前景索引
                # update prospect index
                fg_inds = (labels > 0).nonzero()[:, 0]

            # 计算所需背景数
            # calculate the number of backgrounds needed
            num_bg = self.sample_size - (labels > 0).sum()
            # 如果当前背景数大于所需背景数
            # If the current number of backgrounds is greater than the number of backgrounds needed,
            if len(bg_inds) > num_bg:
                # torch.randint在0到len(bg_inds)之间，随机产生size为(num_bg,)的数组
                # Use the torch.randint function to generate a random array of size num_bg between 0 and len(bg_inds).
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                # 将enable_inds标签置为0
                # set the label of enable_inds to 0
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            # 如果该类别没有GT的话，将该类别的全部label置为0，即认为所有anchor都是背景类别
            # If there is no GT in the category, set all labels in the category to 0,
            # indicating that all anchors are background categories.
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                # anchor和GT的IoU小于unmatch_threshold的anchor的类别设置为背景类别
                # set the category of the anchor with GT's IoU less than unmatch_threshold to the background category
                labels[bg_inds] = 0
                # 将前景赋给对应类别
                # assign the corresponding category to the foreground
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        # 计算bbox_targets和teg_weight
        # 初始化每个anchor的7个回归参数，均置为0
        # calculate bbox_targets and reg_weights
        # Initialize the 7 regression parameters of each anchor and set them to 0.
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        # 如果该帧中有该类别的GT索引，需要对设置为正样本类别的anchor进行编码操作
        # If there are GTs of that category in the processing frame,
        # encode those anchors that are set to the positive sample category.
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # 使用anchor_to_gt_argmax[fg_inds]来重复索引每个anchor对应前景的gt_boxes
            # use anchor_to_gt_argmax[fg_inds] to iteratively index the GT_box in the foreground corresponding to each anchor
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            # 提取所有属于前景的anchor
            # extract all anchors belonging to the foreground
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        # 初始化回归权重并置为0
        # initialize the regression weights and set them to zero
        reg_weights = anchors.new_zeros((num_anchors,))

        # PointPillars回归权重中不需要norm_by_num_examples
        # norm_by_num_examples is not required in the regression weights of PointPillars.
        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            # 将前景anchor回归权重置为1
            # set the regression weight of foreground anchor to 1
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
