import os
import numpy as np
from skimage import io
from sklearn.metrics import confusion_matrix

# ---------------------- Accuracy Metrics Utility Functions ----------------------

def check_size(eval_segm, gt_segm):
    if eval_segm.shape != gt_segm.shape:
        raise ValueError("Shape mismatch: eval_segm and gt_segm must have the same shape.")

def extract_classes(segm):
    cl = np.unique(segm)
    return cl, len(cl)

def extract_masks(segm, cl, n_cl):
    h, w = segm.shape
    masks = np.zeros((n_cl, h, w), dtype=bool)
    for i, c in enumerate(cl):
        masks[i] = segm == c
    return masks

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    return extract_masks(eval_segm, cl, n_cl), extract_masks(gt_segm, cl, n_cl)

def pixel_accuracy(eval_segm, gt_segm):
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    sum_n_ii = np.sum([np.logical_and(eval_mask[i], gt_mask[i]).sum() for i in range(n_cl)])
    sum_t_i  = np.sum([gt_mask[i].sum() for i in range(n_cl)])
    return sum_n_ii / sum_t_i if sum_t_i != 0 else 0

def mean_accuracy(eval_segm, gt_segm):
    check_size(eval_segm, gt_segm)
    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    accuracy = []
    for i in range(n_cl):
        n_ii = np.logical_and(eval_mask[i], gt_mask[i]).sum()
        t_i  = gt_mask[i].sum()
        accuracy.append(n_ii / t_i if t_i != 0 else 0)
    return np.mean(accuracy)

def mean_IU(eval_segm, gt_segm):
    check_size(eval_segm, gt_segm)
    cl = np.union1d(np.unique(eval_segm), np.unique(gt_segm))
    n_cl = len(cl)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    IU = []
    for i in range(n_cl):
        n_ii = np.logical_and(eval_mask[i], gt_mask[i]).sum()
        t_i = gt_mask[i].sum()
        n_ij = eval_mask[i].sum()
        if t_i + n_ij - n_ii > 0:
            IU.append(n_ii / (t_i + n_ij - n_ii))
    return np.mean(IU) if IU else 0

def frequency_weighted_IU(eval_segm, gt_segm):
    check_size(eval_segm, gt_segm)
    cl = np.union1d(np.unique(eval_segm), np.unique(gt_segm))
    n_cl = len(cl)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)
    total_pixels = gt_segm.size
    fwIU = 0
    for i in range(n_cl):
        n_ii = np.logical_and(eval_mask[i], gt_mask[i]).sum()
        t_i = gt_mask[i].sum()
        n_ij = eval_mask[i].sum()
        if t_i + n_ij - n_ii > 0:
            fwIU += (t_i * n_ii) / (t_i + n_ij - n_ii)
    return fwIU / total_pixels if total_pixels != 0 else 0

def compute_segmentation_metrics(y_preds, y_trues):
    pix_acc, mean_iou, fw_iou = [], [], []
    for pred, true in zip(y_preds, y_trues):
        pix_acc.append(pixel_accuracy(pred, true))
        mean_iou.append(mean_IU(pred, true))
        fw_iou.append(frequency_weighted_IU(pred, true))
    return {
        "pixel_accuracy": np.mean(pix_acc),
        "mean_IoU": np.mean(mean_iou),
        "frequency_weighted_IoU": np.mean(fw_iou),
    }
