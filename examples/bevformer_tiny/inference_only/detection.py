"""Inference-only BEVFormer-tiny NMS-free detection decoding."""

import torch


POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)


def decode_detections(all_cls_scores, all_bbox_preds, max_num=300,
                      post_center_range=POST_CENTER_RANGE):
    """Return per-batch boxes [N,9], sigmoid scores [N], and class labels [N].

    The box tensor matches ``LiDARInstance3DBoxes.tensor`` used by the
    original Golden recorder: center z is shifted to the bottom center.
    """
    classes = all_cls_scores[-1]
    boxes = all_bbox_preds[-1]
    if classes.ndim != 3 or boxes.ndim != 3 or classes.shape[:2] != boxes.shape[:2]:
        raise ValueError('Expected final decoder outputs [batch, query, channels]')
    num_classes = classes.shape[-1]
    lower = boxes.new_tensor(post_center_range[:3])
    upper = boxes.new_tensor(post_center_range[3:])
    results = []
    for logits, normalized in zip(classes, boxes):
        scores, indices = logits.sigmoid().reshape(-1).topk(max_num)
        labels = indices % num_classes
        selected = normalized[indices // num_classes]
        yaw = torch.atan2(selected[:, 6:7], selected[:, 7:8])
        dimensions = selected[:, 2:4].exp()
        height = selected[:, 5:6].exp()
        decoded = torch.cat((selected[:, 0:2], selected[:, 4:5],
                             dimensions, height, yaw, selected[:, 8:10]), dim=-1)
        mask = ((decoded[:, :3] >= lower).all(-1)
                & (decoded[:, :3] <= upper).all(-1))
        decoded = decoded[mask].clone()
        decoded[:, 2] -= decoded[:, 5] * 0.5
        results.append((decoded, scores[mask], labels[mask]))
    return results
