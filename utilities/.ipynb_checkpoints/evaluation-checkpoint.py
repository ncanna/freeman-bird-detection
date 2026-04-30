'''
Evaluate the predicted bounding box vs the ground truth
'''

import torch
import numpy as np
from detr_detector.util.box_ops import box_cxcywh_to_xyxy
from utilities.coco_utils import coco_annos_to_detr_targets

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [x_min, y_min, x_max, y_max] format.
    Args:
        box1, box2: tensors of shape [4]
    Returns:
        iou: float
    """
    
    # Get the interset portion the the bounding box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1) # Calcualte the area of the intersection
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) # Calcualte the area of the bounding box 1
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) # Calcualte the area of the bounding box 2
    union = area1 + area2 - intersection # Calcualte the area of the union

    return intersection / union if union > 0 else 0.0


def evaluate_map(model, data_loader, device, img_h, img_w,
                 iou_threshold=0.5, confidence_threshold=0.5):
    """
    Compute mAP and IoU for bird detection.

    Args:
        model:                  DETR model
        data_loader:            DataLoader for evaluation (val or test)
        device:                 torch device
        img_h, img_w:           image dimensions
        iou_threshold:          IoU threshold for a detection to be correct (default 0.5)
        confidence_threshold:   minimum confidence score to keep a prediction (default 0.5)

    Returns:
        results: dict with mAP, mean IoU, precision, recall
    """
    from utilities.coco_utils import coco_annos_to_detr_targets

    model.eval()
    all_predictions = []   # list of (confidence, is_correct) per predicted box
    all_gt_count    = 0    # total number of ground truth boxes
    all_ious        = []   # IoU for each matched prediction

    with torch.no_grad():
        for images, annotations in data_loader:
            images  = images.to(device)
            targets = coco_annos_to_detr_targets(annotations, img_h, img_w, device)
            outputs = model(images)

            # get predicted boxes and scores
            # pred_logits shape: [batch, num_queries, num_classes]
            # pred_boxes shape:  [batch, num_queries, 4] in [cx, cy, w, h] normalized
            pred_logits = outputs['pred_logits']   # [batch, 100, 2]
            pred_boxes  = outputs['pred_boxes']    # [batch, 100, 4]

            # convert logits to probabilities
            # take probability of bird class (class 0), excluding background (class 1)
            scores = pred_logits.softmax(-1)[:, :, 0]   # [batch, 100]

            for i in range(len(images)):
                # get predictions for this image above confidence threshold
                img_scores = scores[i]                   # [100]
                img_boxes  = pred_boxes[i]               # [100, 4]

                # filter by confidence threshold
                keep = img_scores > confidence_threshold
                img_scores = img_scores[keep]
                img_boxes  = img_boxes[keep]

                # convert predicted boxes from [cx,cy,w,h] normalized → [x1,y1,x2,y2] pixels
                if len(img_boxes) > 0:
                    img_boxes_xyxy = box_cxcywh_to_xyxy(img_boxes)
                    img_boxes_xyxy[:, [0, 2]] *= img_w
                    img_boxes_xyxy[:, [1, 3]] *= img_h
                else:
                    img_boxes_xyxy = img_boxes

                # get ground truth boxes for this image
                gt_boxes = targets[i]['boxes']           # [num_gt, 4] normalized [cx,cy,w,h]
                all_gt_count += len(gt_boxes)

                if len(gt_boxes) > 0:
                    # convert gt boxes to pixels xyxy
                    gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes)
                    gt_boxes_xyxy[:, [0, 2]] *= img_w
                    gt_boxes_xyxy[:, [1, 3]] *= img_h
                else:
                    gt_boxes_xyxy = gt_boxes

                # match predictions to ground truth using IoU
                gt_matched = set()
                for score, pred_box in sorted(
                        zip(img_scores.tolist(), img_boxes_xyxy.tolist()),
                        key=lambda x: -x[0]):           # sort by confidence descending

                    best_iou  = 0.0
                    best_gt   = -1
                    for gt_idx, gt_box in enumerate(gt_boxes_xyxy.tolist()):
                        if gt_idx in gt_matched:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt  = gt_idx

                    is_correct = best_iou >= iou_threshold and best_gt not in gt_matched
                    if is_correct:
                        gt_matched.add(best_gt)
                        all_ious.append(best_iou)

                    all_predictions.append((score, int(is_correct)))

    # compute precision-recall curve
    if len(all_predictions) == 0:
        return {'mAP': 0.0, 'mean_iou': 0.0, 'precision': 0.0, 'recall': 0.0}

    # sort by confidence descending
    all_predictions.sort(key=lambda x: -x[0])
    confidences = [p[0] for p in all_predictions]
    is_correct  = [p[1] for p in all_predictions]

    # cumulative TP and FP
    tp_cumsum = np.cumsum(is_correct)
    fp_cumsum = np.cumsum([1 - c for c in is_correct])

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls    = tp_cumsum / (all_gt_count + 1e-6)

    # compute AP using trapezoidal rule
    ap = np.trapz(precisions, recalls)

    results = {
        'mAP':       round(float(ap) * 100, 2),          # as percentage
        'mean_iou':  round(float(np.mean(all_ious)) * 100, 2) if all_ious else 0.0,
        'precision': round(float(precisions[-1]) * 100, 2),
        'recall':    round(float(recalls[-1]) * 100, 2),
        'num_predictions': len(all_predictions),
        'num_gt_boxes':    all_gt_count
    }

    return results