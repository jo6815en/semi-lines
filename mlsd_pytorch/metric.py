import cv2
import numpy as np


def F1_score_128(pred_lines_128_list, gt_lines_128_list, thickness=3):
    """
     @brief heat  F1 score, draw the lines to a 128 * 128 img
     @pred_lines_128 [ [x0, y0, x1, y1],  ... ]
     @gt_lines_128_list [ [x0, y0, x1, y1],  ... ]
    """
    pred_heatmap = np.zeros((128, 128), np.uint8)
    gt_heatmap = np.zeros((128, 128), np.uint8)

    for l in pred_lines_128_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(pred_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    for l in gt_lines_128_list:
        x0, y0, x1, y1 = l
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))
        cv2.line(gt_heatmap, (x0, y0), (x1, y1), (1, 1, 1), thickness, 8)

    pred_heatmap = np.array(pred_heatmap, np.float32)
    gt_heatmap = np.array(gt_heatmap, np.float32)

    intersection = np.sum(gt_heatmap * pred_heatmap)
    # union = np.sum(gt_heatmap) + np.sum(gt_heatmap)
    eps = 0.001
    # dice = (2. * intersection + eps) / (union + eps)

    recall = intersection /(np.sum(gt_heatmap) + eps)
    precision = intersection /(np.sum(pred_heatmap) + eps)

    fscore = (2 * precision * recall) / (precision + recall + eps)
    return fscore, recall, precision


def msTPFP(line_pred, line_gt, threshold):
    line_pred = line_pred.reshape(-1, 2, 2)[:, :, ::-1]
    line_gt = line_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )

    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), bool)
    tp = np.zeros(len(line_pred), float)
    fp = np.zeros(len(line_pred), float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def msTPFP_tampered(line_pred, line_gt, threshold):
    threshold_x = 50
    threshold_y = 150
    line_pred = line_pred.reshape(-1, 2, 2)[:, :, ::-1]
    line_gt = line_gt.reshape(-1, 2, 2)[:, :, ::-1]
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)

    should_flip = np.argmin(np.stack([
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]]), axis=0)

    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0])

    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), bool)
    tp = np.zeros(len(line_pred), float)
    fp = np.zeros(len(line_pred), float)
    hit2 = np.zeros(len(line_gt), bool)
    tp2 = np.zeros(len(line_pred), float)
    fp2 = np.zeros(len(line_pred), float)

    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1

    for i in range(len(line_pred)):
        x0, x1 = line_pred[i, :, 1]
        y0, y1 = line_pred[i, :, 0]
        x0_gt, x1_gt = line_gt[choice[i], :, 1]
        y0_gt, y1_gt = line_gt[choice[i], :, 0]
        if should_flip[i, choice[i]]:
            x0_gt, x1_gt = x1_gt, x0_gt
            y0_gt, y1_gt = y1_gt, y0_gt
        dist_x = (x0 - x0_gt) ** 2 + (x1 - x1_gt) ** 2
        dist_y = (y0 - y0_gt) ** 2 + (y1 - y1_gt) ** 2

        if dist_x < threshold_x and not hit2[choice[i]]:
            hit2[choice[i]] = True
            tp2[i] = 1
        else:
            fp2[i] = 1

    return tp2, fp2


def TPFP(lines_dt, lines_gt, threshold):
    lines_dt = lines_dt.reshape(-1,2,2)[:,:,::-1]
    lines_gt = lines_gt.reshape(-1,2,2)[:,:,::-1]
    diff = ((lines_dt[:, None, :, None] - lines_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff,1)
    dist = np.min(diff,1)
    hit = np.zeros(len(lines_gt), bool)
    tp = np.zeros(len(lines_dt), float)
    fp = np.zeros(len(lines_dt), float)

    for i in range(lines_dt.shape[0]):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def AP(tp, fp):
    recall = tp
    precision = tp/np.maximum(tp+fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]

    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

    return ap