import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        # Three prior frames of different scales exist for each point,
        # and two directions (0 and 90 degrees) exist for each prior frame.
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # If there is directional loss, then add a directional convolution layer.
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    # initialize parameters
    def init_weights(self):
        pi = 0.01
        # initialize the bias of the classification convolution
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # initialize the weights of the classification convolution
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # Get the information from the dictionary after backbone processing.
        spatial_features_2d = data_dict['spatial_features_2d']

        # For each coordinate point, there exist 6 category predictions for the prior frame.
        cls_preds = self.conv_cls(spatial_features_2d)
        # For each coordinate point there exist 6 parameter predictions for the a priori frame.
        # Each a priori box needs to predict 7 parameters.
        box_preds = self.conv_box(spatial_features_2d)

        # Adjust the dimension,
        # i.e. swap the data information from the category or parameter adjustment dimension to the last dimension.
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        # Put the predictions of category and prior frame adjustments into the forward propagation dictionary.
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        # perform prediction of direction classification
        if self.conv_dir_cls is not None:
            # The prediction for each prior frame should be in one of two directions.
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            # Adjust the dimension,
            # i.e. swap the data information from the category or parameter adjustment dimension to the last dimension.
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            # Put the direction prediction results into the forward propagation dictionary.
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            # Add the results of the GT assignment to the forward propagation dictionary.
            self.forward_ret_dict.update(targets_dict)

        # If it is not a training model, then box prediction results can be generated directly.
        if not self.training or self.predict_boxes_when_training:
            # Decode and generate the final result based on the prediction results.
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
