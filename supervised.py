import argparse
import logging
import os
import random

import torch
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler


from dataset.semi import SemiDataset, SemiDataset_collate_fn
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

from mlsd_pytorch.metric import F1_score_128, TPFP, msTPFP, AP, msTPFP_tampered
from mlsd_pytorch.utils.decode import deccode_lines_TP
from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.loss import LineSegmentLoss
import tqdm
from mlsd_pytorch.models.build_model import build_model

from pathlib import Path


parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--prev_best', type=str, default='', required=False)
parser.add_argument('--is_wireframe', type=bool, default=False, required=False)

__mapping_dataset_collate_fn = {
    'wireframe': SemiDataset_collate_fn,
}

root = Path(__file__).parent.parent  # top-level directory
TRAINING_PATH = root / "UniMatch/"


def get_collate_fn(cfg):
    if cfg.datasets.name not in __mapping_dataset_collate_fn.keys():
        raise NotImplementedError('Dataset Type not supported!')
    return __mapping_dataset_collate_fn[cfg.datasets.name]


def evaluate(model, loader, cfg, epoch, logger, rank, is_wireframe):
    model.eval()

    thresh = cfg.decode.score_thresh
    topk = cfg.decode.top_k
    min_len = cfg.decode.len_thresh
    sap_thresh = cfg.decode.sap_thresh

    data_iter = tqdm.tqdm(loader, disable=True)
    f_scores = []
    recalls = []
    precisions = []

    input_size = cfg.datasets.input_size

    tp_list, fp_list, scores_list = [], [], []
    n_gt = 0


    with torch.no_grad():
        for batch_data in data_iter:
            imgs = batch_data["xs"].cuda()
            label = batch_data["ys"].cuda()
            batch_outputs = model(imgs)

            batch_outputs = batch_outputs[:, 7:, :, :]

            for outputs, gt_lines_512 in zip(batch_outputs, batch_data["gt_lines_512"]):
                gt_lines_512 = np.array(gt_lines_512, np.float32)

                outputs = outputs.unsqueeze(0)

                center_ptss, pred_lines, _, scores = \
                    deccode_lines_TP(outputs, thresh, min_len, topk, 3)

                pred_lines = pred_lines.detach().cpu().numpy()
                scores = scores.detach().cpu().numpy()

                pred_lines_128 = 128 * pred_lines / (input_size / 2)

                gt_lines_128 = gt_lines_512 / 4
                fscore, recall, precision = F1_score_128(pred_lines_128.tolist(), gt_lines_128.tolist(),
                                                         thickness=3)
                f_scores.append(fscore)
                recalls.append(recall)
                precisions.append(precision)

                if is_wireframe:
                    tp, fp = msTPFP(pred_lines_128, gt_lines_128, sap_thresh)
                else:
                    tp, fp = msTPFP_tampered(pred_lines_128, gt_lines_128, sap_thresh)

                n_gt += gt_lines_128.shape[0]
                tp_list.append(tp)
                fp_list.append(fp)
                scores_list.append(scores)

        f_score = np.array(f_scores, np.float32).mean()
        recall = np.array(recalls, np.float32).mean()
        precision = np.array(precisions, np.float32).mean()

        tp_list = np.concatenate(tp_list)
        fp_list = np.concatenate(fp_list)
        scores_list = np.concatenate(scores_list)
        idx = np.argsort(scores_list)[::-1]
        tp = np.cumsum(tp_list[idx]) / n_gt
        fp = np.cumsum(fp_list[idx]) / n_gt
        sAP = AP(tp, fp) * 100
        if rank == 0:
            logger.info("Step: {:}, f_score: {:.5f}, recall: {:.2f}, precision: {:.2f}, sAP10: {:.2f}\n ".
                        format(epoch, f_score, recall, precision, sAP))

        return {
            'fscore': f_score,
            'recall': recall,
            'precision': precision,
            'sAP10': sAP
        }


def main():
    args = parser.parse_args()
    cfg = get_cfg_defaults()

    output_dir = Path(TRAINING_PATH, args.save_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    print("Here is output_dir: ", output_dir)

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    cfg.merge_from_file(args.config)

    loss_fn = LineSegmentLoss(cfg)

    logger = init_log('global', logging.INFO)
    logger.propagate = 1

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    # model = DeepLabV3Plus(cfg)
    model = build_model(cfg).cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay)
    
    #StepLR (Decay the learning rate by gamma every step_size epochs)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    if os.path.exists(cfg.train.load_from):
        print('load from: ', cfg.train.load_from)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(cfg.train.load_from, map_location=device), strict=False)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
    #                                                  output_device=local_rank, find_unused_parameters=False)

    trainset = SemiDataset(cfg, 'train_l', args.labeled_id_path)
    valset = SemiDataset(cfg, 'val', is_train=False)

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size,
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler,
                             collate_fn=get_collate_fn(cfg))
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler, collate_fn=get_collate_fn(cfg))

    iters = 0
    total_iters = len(trainloader) * cfg.train.num_train_epochs
    previous_best = 0.0
    prev_sAP = 0.0
    epoch = -1
    best_epoch = 0

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        prev_sAP = checkpoint['previous_sAP']
        best_epoch = checkpoint['best_epoch']
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    elif args.prev_best:
        checkpoint = torch.load(os.path.join(args.prev_best, 'best_sAP.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if rank == 0:
            logger.info('****** Load pretrained weights ******')


    for epoch in range(epoch + 1, cfg.train.num_train_epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best fscore: {:.2f}, Previous sAP10: {:.2f}, epoch: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, prev_sAP, best_epoch))

        model.train()
        total_loss = AverageMeter()

        step_n = 0

        train_avg_loss = AverageMeter()
        train_avg_center_loss = AverageMeter()
        train_avg_replacement_loss = AverageMeter()
        train_avg_line_seg_loss = AverageMeter()
        train_avg_junc_seg_loss = AverageMeter()
        train_avg_match_loss = AverageMeter()
        train_avg_match_rario = AverageMeter()
        train_avg_t_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, batch in enumerate(trainloader):
            imgs = batch["xs"].cuda()
            label = batch["ys"].cuda()
            outputs = model(imgs)
            loss_dict = loss_fn(outputs, label, batch["gt_lines_tensor_512_list"],
                                batch["sol_lines_512_all_tensor_list"])
            loss = loss_dict['loss']

            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_avg_loss.update(loss.item(), 1)

            train_avg_center_loss.update(loss_dict['center_loss'].item(), 1)
            train_avg_replacement_loss.update(loss_dict['displacement_loss'].item(), 1)
            train_avg_line_seg_loss.update(loss_dict['line_seg_loss'].item(), 1)
            train_avg_junc_seg_loss.update(loss_dict['junc_seg_loss'].item(), 1)
            train_avg_match_loss.update(float(loss_dict['match_loss']), 1)
            train_avg_match_rario.update(loss_dict['match_ratio'], 1)

            iters = epoch * len(trainloader) + step_n

            if len(trainloader) > 0 and (i % max(1, (len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, avg = {:.3f}, c: {:.3f}, d: {:.3f}, l: {:.3f}, ' \
                            'junc:{:.3f}, m:{:.3f}, m_r:{:.2f} '.format(
                    i,
                    loss.item(),
                    train_avg_loss.avg,
                    train_avg_center_loss.avg,
                    train_avg_replacement_loss.avg,
                    train_avg_line_seg_loss.avg,
                    train_avg_junc_seg_loss.avg,
                    train_avg_match_loss.avg,
                    train_avg_match_rario.avg))

            step_n += 1

        if epoch % 1 == 0:
            m = evaluate(model, valloader, cfg, epoch, logger, rank, is_wireframe=args.is_wireframe)

            fscore = m['fscore']
            sAP = m["sAP10"]

            is_best = fscore > previous_best
            is_best_sAP = sAP > prev_sAP
            previous_best = max(fscore, previous_best)
            prev_sAP = max(sAP, prev_sAP)
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'previous_best': previous_best,
                    'previous_sAP': prev_sAP,
                }
                torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
                if is_best_sAP:
                    best_epoch = epoch
                    torch.save(checkpoint, os.path.join(args.save_path, 'best_sAP.pth'))
                if is_best:
                    torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        # scheduler.step()


if __name__ == '__main__':
    main()
