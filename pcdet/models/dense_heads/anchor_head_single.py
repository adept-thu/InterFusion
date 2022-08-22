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

        # 每个点均有3个尺度的先验框，每个先验框均有2个方向（即0度和90度）
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

        # 如果存在方向损失，可以添加方向卷积层Conv2d（512,12,kernel_size=(1,1),stride=(1,1))
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

    # 初始化参数
    # initialize parameters
    def init_weights(self):
        pi = 0.01
        # 初始化分类卷积的偏置
        # initialize the bias of the classification convolution
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # 初始化分类卷积的权重
        # initialize the weights of the classification convolution
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # 从字典中获取经过backbone处理过的信息
        # Get the information from the dictionary after backbone processing.
        spatial_features_2d = data_dict['spatial_features_2d']

        # 每个坐标点上面6个先验框的类别预测
        # For each coordinate point, there exist 6 category predictions for the prior frame.
        cls_preds = self.conv_cls(spatial_features_2d)
        # 每个坐标点上面6个先验框的参数预测
        # 其中每个先验框需要预测7个参数，分别为(x,y,z,w,l,h,θ)
        # For each coordinate point there exist 6 parameter predictions for the a priori frame.
        # Each a priori box needs to predict 7 parameters.
        box_preds = self.conv_box(spatial_features_2d)

        # 调整维度，将类别/参数放置到最后一个维度
        # Adjust the dimension,
        # i.e. swap the data information from the category or parameter adjustment dimension to the last dimension.
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        # 将类别和先验框调整预测结果放入前向传播字典中
        # Put the predictions of category and prior frame adjustments into the forward propagation dictionary.
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        # 进行方向分类预测
        # make prediction of direction classification
        if self.conv_dir_cls is not None:
            # 每个先验框都要预测为两个方向中的一个
            # The prediction for each prior frame should be in one of two directions.
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            # 将类别和先验框方向预测结果放入最后一个维度中
            # Adjust the dimension,
            # i.e. swap the data information from the category or parameter adjustment dimension to the last dimension.
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            # 将方向预测的结果放入前向传播的字典中
            # Put the direction prediction results into the forward propagation dictionary.
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            # 将GT分配结果放入前向传播字典中
            # Add the results of the GT assignment to the forward propagation dictionary.
            self.forward_ret_dict.update(targets_dict)

        # 如果不是训练模式，可以直接生成box的预测
        # If it is not a training model, then box prediction results can be generated directly.
        if not self.training or self.predict_boxes_when_training:
            # 根据预测结果解码生成最终结果
            # Decode and generate the final result based on the prediction results.
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
