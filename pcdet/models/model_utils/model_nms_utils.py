import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    # 根据置信度阈值过滤掉大部分置信度低的box，用来提升后面的NMS操作的效率。
    # Most of the boxes with low confidence are filtered out according to the confidence threshold,
    # which is used to improve the efficiency of the later NMS operations.
    src_box_scores = box_scores
    if score_thresh is not None:
        # 获取类别的预测概率大于score_thresh的mask
        # get the category whose predicted probability is greater than score_thresh's mask
        scores_mask = (box_scores >= score_thresh)
        # 根据mask来获取类别的预测值大于score_thresh的anchor的类别
        # get the category whose predicted value is greater than the anchor of score_thresh according to mask
        box_scores = box_scores[scores_mask]
        # 根据mask来获取类别的预测值大于score_thresh的anchor的7个回归参数
        # get the 7 regression parameters of the anchor of the category
        # whose predicted value is greater than score_thresh according to mask
        box_preds = box_preds[scores_mask]

    # 初始化并得到一个空列表，目的是用来存放NMS操作后保留下来的anchor。
    # Initialize and get an empty list, the purpose is to store the anchor retained after NMS operation.
    selected = []
    # 当anchor的类别的预测值大于score_thresh，进行NMS操作，否则返回空。
    # When the predicted value of the anchor's category is greater than score_thresh,
    # perform NMS operation, otherwise return null.
    if box_scores.shape[0] > 0:
        # 仅保留最大的K个anchor的置信度进行NMS操作，K取值为min（nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0])中的较小的值。
        # Only the confidence of the largest K anchors is retained for the NMS operation,
        # and K is taken to be the smaller of min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]).
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))

        # box_scores_nms获取了类别的更新结果。
        # 要想更新box的预测结果，需要调用topk来重新选取并按照从大到小的顺序排列结果，然后再更新box的预测值。
        # box_scores_nms gets the updated results of the category.
        # To update the predicted results of box,
        # you need to call topk to re-pick and sort the results in order from largest to smallest,
        # and then update the predicted values of box.
        boxes_for_nms = box_preds[indices]
        # 调用iou3d_nms_utils的nms_gpu函数来进行NMS操作
        # 将保留下来的box的索引返回，并设置selected_scores = None
        # 再根据返回的索引值获取box的索引值。
        # Call the nms_gpu function of iou3d_nms_utils to perform the NMS operation,
        # return the index of the reserved box and set selected_scores = None,
        # then get the index value of the box according to the returned index value.
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

    if score_thresh is not None:
        # 如果存在置信度阈值，那么scored_mask就是box_scores在src_box_scores中的索引，也就是原始的索引。
        # If a confidence threshold exists,
        # then scored_mask is the index of box_scores in src_box_scores, which is the original index.
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
