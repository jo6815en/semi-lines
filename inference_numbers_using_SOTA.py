import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

import torchvision.transforms.functional as F

from dataset.semi import SemiDataset, SemiDataset_collate_fn
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

from torchvision.utils import save_image
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import tqdm

from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.models.build_model import build_model
from mlsd_pytorch.data import get_train_dataloader, get_val_dataloader
from mlsd_pytorch.metric import F1_score_128, TPFP, msTPFP, AP, msTPFP_tampered
from mlsd_pytorch.utils.decode import deccode_lines_TP
from mlsd_utils import get_pred_lines

from DeepLSD.deeplsd.models.deeplsd_inference import DeepLSD

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--uni_config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--save_images', default='/inference', type=str)
parser.add_argument('--config', type=str, required=True)


__mapping_dataset_collate_fn = {
    'wireframe': SemiDataset_collate_fn,
}


def get_collate_fn(cfg):
    if cfg.datasets.name not in __mapping_dataset_collate_fn.keys():
        raise NotImplementedError('Dataset Type not supported!')
    return __mapping_dataset_collate_fn[cfg.datasets.name]


def point_line_distance(point, line):
    """Calculate the perpendicular distance from a point to a line segment."""
    x0, y0 = point
    x1, y1, x2, y2 = line
    if (x2 - x1) == 0 and (y2 - y1) == 0:
        return np.linalg.norm(point - np.array([x1, y1]))
    return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.linalg.norm([x2 - x1, y2 - y1])


def line_orthogonal_distance(line1, line2):
    """Calculate the orthogonal distance between two line segments."""
    p1, p2 = line1[:2], line1[2:]
    q1, q2 = line2[:2], line2[2:]

    # Perpendicular distances from endpoints of line1 to line2
    d1 = point_line_distance(p1, line2)
    d2 = point_line_distance(p2, line2)

    # Perpendicular distances from endpoints of line2 to line1
    d3 = point_line_distance(q1, line1)
    d4 = point_line_distance(q2, line1)

    # Minimum distance between midpoints
    midpoint1 = (p1 + p2) / 2
    midpoint2 = (q1 + q2) / 2
    d5 = np.linalg.norm(midpoint1 - midpoint2)

    return min(d1, d2, d3, d4, d5)


def line_distance(line1, line2):
    """Calculate a simple Euclidean distance between two lines."""
    dist1 = np.linalg.norm(line1[:2] - line2[:2])
    dist2 = np.linalg.norm(line1[2:] - line2[2:])
    return dist1 + dist2


def calculate_confidence_scores(detected_lines, gt_lines):
    """Calculate confidence scores for detected lines based on ground truth lines."""
    scores = []
    for d_line in detected_lines:
        max_score = 0
        for gt_line in gt_lines:
            score = 1 / (line_distance(d_line, gt_line) + 1e-6)  # Add small value to avoid division by zero
            max_score = max(max_score, score)
        scores.append(max_score)
    return np.array(scores)


def inference(model, loader, cfg, is_wireframe=False, save_img=None):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    thresh = cfg.decode.score_thresh
    topk = cfg.decode.top_k
    min_len = cfg.decode.len_thresh
    sap_thresh = cfg.decode.sap_thresh

    data_iter = tqdm.tqdm(loader)
    f_scores = []
    recalls = []
    precisions = []

    input_size = cfg.datasets.input_size

    tp_list, fp_list, scores_list = [], [], []
    n_gt = 0

    count = 1

    if save_img:
        os.makedirs(save_img, exist_ok=True)

    with torch.no_grad():
        for batch_data in data_iter:
            imgs = batch_data["xs"].cuda()
            label = batch_data["ys"].cuda()

            img_org = batch_data["origin_imgs"]


            # print("Before running the model!", imgs.shape, imgs.type)
            #batch_outputs = model(imgs)

            # This is from the mlsd_pytorch code
            #batch_outputs = batch_outputs[:, 7:, :, :]

            #Finnwood:
            img_name = batch_data["img_fns"][0].split("/")[5]
            #Samseg / snoge  och wireframe
            #img_name = batch_data["img_fns"][0].split("/")[4]
            #Skrylle
            #img_name = batch_data["img_fns"][0].split("/")[4][6:]

            if is_wireframe:
                orig_image = cv2.imread("./data/wireframe_raw/images/" + img_name)
                height, width, _ = orig_image.shape
            else:
                height, width = 720, 1280

            detections = np.load("lines/letr_lines_finnwoods/val/letr_" + img_name[:-3] + "npy")

            letr = True
            if letr:
                detections = detections.reshape(-1, 2, 2)
                detections = detections[:, :, [1, 0]]

            detections = np.expand_dims(detections, axis=0)

            detections[:, :, :, 1] = detections[:, :, :, 1] * (512 / height)
            detections[:, :, :, 0] = detections[:, :, :, 0] * (512 / width)

            count = count + 1
            for outputs, gt_lines_512 in zip(detections, batch_data["gt_lines_512"]):
                gt_lines_512 = np.array(gt_lines_512, np.float32)

                #outputs = outputs.unsqueeze(0)

                outputs = torch.tensor(outputs)

                #thresh = 0.08

                # print("Outputs in for loop: ", outputs.shape)
                # print("gt_lines_512: ", gt_lines_512.shape)

                #center_ptss, pred_lines, _, scores = \
                #    deccode_lines_TP(outputs, thresh, min_len, topk, 3)

                #print("Scores from TP: ", scores)

                # Extract start and end points from outputs
                start_point = outputs[:, 0, :]  # shape: (2, 2)
                end_point = outputs[:, 1, :]  # shape: (2, 2)

                pred_lines = torch.cat((start_point, end_point), dim=-1)

                pred_lines = pred_lines.detach().cpu().numpy()
                #scores = scores.detach().cpu().numpy()

                #pred_lines_128 = 128 * pred_lines / (input_size / 2)
                pred_lines_128 = pred_lines / 4
                p_lines = pred_lines_128

                gt_lines_128 = gt_lines_512 / 4

                if letr:
                    scores = np.load("lines/letr_lines_finnwoods/val/letrscores_" + img_name[:-3] + "npy").squeeze(0)
                    # print("Scores from LETR: ", scores)
                else:
                    # scores = torch.ones(pred_lines.shape[0])
                    scores = calculate_confidence_scores(pred_lines_128, gt_lines_128)

                fscore, recall, precision = F1_score_128(pred_lines_128.tolist(), gt_lines_128.tolist(),
                                                         thickness=3)
                f_scores.append(fscore)
                recalls.append(recall)
                precisions.append(precision)

                if is_wireframe:
                    tp, fp = msTPFP(pred_lines_128, gt_lines_128, sap_thresh)
                else:
                    tp, fp = msTPFP_tampered(pred_lines_128, gt_lines_128, sap_thresh)

                should_plot = False
                if should_plot and count % 10 == 0:
                    # Create a figure with two subplots
                    #fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                    img_temp1 = np.copy(img_org)

                    img_temp2 = np.copy(img_org)
                    img_temp3 = np.copy(img_org)

                    # for l in gt_lines_512:
                    #    cv2.line(img_temp2[0], (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 200, 200), 1, 16)

                    # Plot ground truth lines with true positives and false positives highlighted
                    for i, l in enumerate(p_lines):
                        if i == len(tp):
                            break
                        if tp[i] == 1:
                            cv2.line(img_temp2[0], (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 165, 0), 1, 16)  # Mark true positives in green (0, 215, 0)
                        elif fp[i] == 1:
                            cv2.line(img_temp2[0], (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 165, 0), 1, 16)  # Mark false positives in red (215, 0, 0)

                    # Plot predicted lines
                    for l in p_lines:
                    #for l in gt_lines_512:
                        cv2.line(img_temp3[0], (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 160, 0), 1, 16)

                    #axes[0].imshow(img_temp2[0])
                    #axes[1].imshow(img_temp3[0])

                    # Set titles for the subplots
                    #axes[0].set_title('Predicted Lines')
                    #axes[1].set_title('Ground Truth')

                    # Display the combined plot
                    #plt.show()

                    #plt.imshow(img_temp2[0])
                    #plt.axis('off')
                    #plt.show()

                    plt.imshow(img_temp3[0])
                    plt.axis('off')
                    plt.show()

                n_gt += gt_lines_128.shape[0]
                tp_list.append(tp)
                fp_list.append(fp)
                scores_list.append(scores)

        f_score = np.array(f_scores, np.float32).mean()
        f_median = np.median(np.array(f_scores, np.float32), axis=0)
        print("f_median: ", f_median)
        recall = np.array(recalls, np.float32).mean()
        precision = np.array(precisions, np.float32).mean()

        tp_list = np.concatenate(tp_list)
        fp_list = np.concatenate(fp_list)
        scores_list = np.concatenate(scores_list)
        idx = np.argsort(scores_list)[::-1]
        tp = np.cumsum(tp_list[idx]) / n_gt
        fp = np.cumsum(fp_list[idx]) / n_gt
        rcs = tp
        pcs = tp / np.maximum(tp + fp, 1e-9)
        sAP = AP(tp, fp) * 100
        print("==> f_score: {}, recall: {}, precision:{}, sAP10: {}".format(f_score, recall, precision, sAP))

        return {
            'fscore': f_score,
            'recall': recall,
            'precision': precision,
            'sAP10': sAP
        }

def main():
    torch.cuda.empty_cache()
    args = parser.parse_args()

    # cfg = yaml.load(open(args.uni_config, "r"), Loader=yaml.Loader)

    cfg = get_cfg_defaults()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    cfg.merge_from_file(args.config)

    cudnn.enabled = True
    cudnn.benchmark = True

    dist.init_process_group(backend='nccl', init_method='env://')

## Org. model
    model = build_model(cfg).cuda()

    if os.path.exists(cfg.train.load_from):
        print('load from: ', cfg.train.load_from)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(cfg.train.load_from, map_location=device), strict=False)

    print("Loaded the model")
    local_rank = int(os.environ["LOCAL_RANK"])

    checkpoint = torch.load(os.path.join(args.save_path, 'best_sAP.pth'))
    model.load_state_dict(checkpoint['model'])

    print("Checkpoint: ", os.path.join(args.save_path, 'best_sAP.pth'))


    is_wireframe = False
    #valset = SemiDataset(cfg, 'val', is_train=False)
    #valset = SemiDataset(cfg, 'test', is_train=False)
    #valset = SemiDataset(cfg, 'val_sam_segs', is_train=False)
    #valset = SemiDataset(cfg, 'val_skrylle', is_train=False)
    valset = SemiDataset(cfg, 'test_finn', is_train=False)

    #valset = SemiDataset(cfg, 'test_wireframe', is_train=False)
    #is_wireframe = True
    #valset = SemiDataset(cfg, 'test_finn_with_wireframe', is_train=False)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler, collate_fn=get_collate_fn(cfg))

    # mlsd_val_loader = get_val_dataloader(mlsd_cfg)

    m = inference(model, valloader, cfg, is_wireframe, save_img='inference')

    print("DONE!")


if __name__ == '__main__':
    main()
