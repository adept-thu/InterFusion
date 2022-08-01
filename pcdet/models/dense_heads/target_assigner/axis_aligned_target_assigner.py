import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class AxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        # anchor generates configuration parameters
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        # find the parameters of the corresponding anchor to predict the box
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        # code the 7 residual parameters of the box
        self.box_coder = box_coder
        # When specifying positive and negative samples,
        # it is considered that the IoU of GT and a priori frame is calculated based on the BEV perspective,
        # so it is not necessary to consider the matching of the height in the z-axis direction.
        self.match_height = match_height
        # get the type names['Car','Pedestrian','Cyclist']
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        # set the sampling coefficients for foreground and background
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        # set the total number of samples
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        # set the weight of the foreground
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        # Set the threshold when the IoU match of the category is a positive sample.
        # {'Car:0.6','Pedestrian:0.5','Cyclelist:0.5'}
        self.matched_thresholds = {}
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
        # Initialize the result list and extract the corresponding gt_box and category.
        bbox_targets = []
        cls_labels = []
        reg_weights = []

        # get batch size
        batch_size = gt_boxes_with_classes.shape[0]
        # get all categories of GT
        gt_classes = gt_boxes_with_classes[:, :, -1]
        # get all 7 box parameters of GT
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        # Match the foreground and background of the anchor frame by frame for all data in the batch.
        for k in range(batch_size):
            # get the gt_boxes of the current frame
            cur_gt = gt_boxes[k]
            # get the maximum number of data center GTs in the same batch
            cnt = cur_gt.__len__() - 1
            # Find the last non-zero box,
            # if the number of non-zero boxes is insufficient,
            # the missing box will be automatically filled with 0.
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            # extract the non-zero box and category of the current frame
            cur_gt = cur_gt[:cnt + 1]
            # convert data types
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            target_list = []
            # The anchor and GT are classified for each frame,
            # while the front background is calculated separately.
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                # anchor_class_name: Car | Pedestrian | Cyclist
                if cur_gt_classes.shape[0] > 1:
                    # self.class_names : ['Car','Pedestrian','Cyclist']
                    # In order to get the same minecrumbs belonging to the GT in the list as the current processed category,
                    # the list index is subtracted by 1 before processing, and finally the category mask is obtained.
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

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
                    # calculate the required variables and get the feature map size
                    feature_map_size = anchors.shape[:3]
                    # flatten all the anchors
                    anchors = anchors.view(-1, anchors.shape[-1])
                    # Get the current category to be processed in the frame based on the categorymask index.
                    # Car | Pedestrian | Cyclist
                    selected_classes = cur_gt_classes[mask]

                # Use assign_targets_single to assign gt_boxes independently for a category of anchor.
                # Set coding and regression weights for foreground and background boxes.
                single_target = self.assign_targets_single(
                    anchors,                                                            # all the anchors of the category
                    cur_gt[mask],                                                       # GT_box
                    gt_classes=selected_classes,                                        # the currently selected category
                    # the threshold of the current category of anchor matching with GT as a positive sample
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    # the threshold of the current category of anchor matching with GT as a negative sample
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)
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

            # Add the obtained results to the corresponding dictionaries in sequence.
            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
            # Up to this position, all point clouds for a given batch have been processed.

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

        # Initialize the anchor's corresponding label and gt_id and set them to -1. 
        # The loss will not be considered in the calculation and the category of the background will be set to 0.
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        # calculate the anchor in the category
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # calculate the IoU between the GT of a category and the corresponding anchor in the processing frame
            # anchor_by_gt_overlap: all anchors in the current category, and IoU of all GTs in the current category
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            # The speed of these two versions depends the environment and the number of anchors.
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()

            # Get each anchor, and the maximum IoU of the GT corresponding to that anchor.
            # anchor_to_gt_argmax: the dimension of the data is equal to the length of the obtained anchor and the index is GT.
            anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            # anchor_to_gt_max: get the IoU value of the GT that best matches each anchor
            anchor_to_gt_max = anchor_by_gt_overlap[
                torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax
            ]

            # Get the index and IoU value of the most matching anchor for each GT.
            # get the index of the most matching anchor for each GT
            gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            # get the IoU value of the most matching anchor for each GT
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            # set -1 to the IoU value of the anchor that does not match in GT
            # the mask of the GT of the anchor that did not match
            empty_gt_mask = gt_to_anchor_max == 0
            # Set the IoU value of the GT of the anchor that is not matched to -1
            gt_to_anchor_max[empty_gt_mask] = -1

            # Get the index of the anchor in the anchor that has the maximum IoU with GT, i.e., the foreground anchor.
            # nonzero: get the position of the non-zero element in the arrayarray (array index)
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            # get the most matching anchor and the corresponding GT index in that category
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            # add the category of GT to the label of the corresponding anchor
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            # add the index of GT to the gt_ids of the corresponding anchor
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()

            # calculate indexes of foreground and background, update label and gt_ids
            # Get the mask with IoU greater than the given threshold in the most matching anchor.
            pos_inds = anchor_to_gt_max >= matched_threshold
            # Get the index of the GT with IoU greater than the given threshold in the most matching anchor.
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            # Add the category of the GT corresponding to the pos_anchor to the label of the corresponding anchor.
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            # Add the index of the GT corresponding to the pos_anchor to the gt_id of the corresponding anchor.
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            # get the index of the background anchor
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        # get the index of the foreground anchor
        # get the IoU value between unmatched_threshold and matched_threshold of the foreground anchor
        fg_inds = (labels > 0).nonzero()[:, 0]
        # Up to this position, it has been confirmed whether part of the anchor belongs to the foreground or the background.

        # filter and assign values to the foreground and background of the anchor
        # If there is a foreground sampling ratio,
        # the foreground and background anchors need to be sampled separately.
        # This operation is not available in PointPillars,
        # because the focal_loss function is used for foreground and background.
        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            # if the number of anchors in the foreground is greater than the number of sampled foregrounds,
            if len(fg_inds) > num_fg:
                # calculate the number of foreground anchors to be dropped
                num_disabled = len(fg_inds) - num_fg
                # Generate random index values among the number of foregrounds,
                # then get the first num_disabled indexes.
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                # set -1 to the IoU of the anchor that was dropped
                labels[disable_inds] = -1
                # update prospect index
                fg_inds = (labels > 0).nonzero()[:, 0]

            # calculate the number of backgrounds needed
            num_bg = self.sample_size - (labels > 0).sum()
            # If the current number of backgrounds is greater than the number of backgrounds needed,
            if len(bg_inds) > num_bg:
                # Use the torch.randint function to generate a random array of size num_bg between 0 and len(bg_inds).
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                # set the label of enable_inds to 0
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            # If there is no GT in the category, set all labels in the category to 0,
            # indicating that all anchors are background categories.
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
            else:
                # set the category of the anchor with GT's IoU less than unmatched_threshold to the background category
                labels[bg_inds] = 0
                # assign the corresponding category to the foreground
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

        # calculate bbox_targets and reg_weights
        # Initialize the 7 regression parameters of each anchor and set them to 0.
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        # If there are GTs of that category in the processing frame,
        # encode those anchors that are set to the positive sample category.
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            # use anchor_to_gt_argmax[fg_inds] to iteratively index the GT_box in the foreground corresponding to each anchor
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            # extract all anchors belonging to the foreground
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        # initialize the regression weights and set them to zero
        reg_weights = anchors.new_zeros((num_anchors,))

        # norm_by_num_examples is not required in the regression weights of PointPillars.
        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0   # set the regression weight of foreground anchor to 1

        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
        }
        return ret_dict
