import argparse
import logging
import os
import pprint
import random

import numpy as np

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import yaml

from dataset.semi import SemiDataset, SemiDataset_collate_fn
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

from mlsd_pytorch.metric import F1_score_128, TPFP, msTPFP, AP
from mlsd_pytorch.utils.decode import deccode_lines_TP
from mlsd_pytorch.data import get_train_dataloader, get_val_dataloader
from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.loss import LineSegmentLoss
import tqdm
from mlsd_pytorch.models.build_model import build_model


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--prev_best', type=str, default='', required=False)
parser.add_argument('--is_wireframe', type=bool, default=False, required=False)



__mapping_dataset_collate_fn = {
    'wireframe': SemiDataset_collate_fn,
}


def get_collate_fn(cfg):
    if cfg.datasets.name not in __mapping_dataset_collate_fn.keys():
        raise NotImplementedError('Dataset Type not supported!')
    return __mapping_dataset_collate_fn[cfg.datasets.name]


def count_true(matrix):
    true_count = 0
    for row in matrix:
        for element in row:
            if element:  # Check if element is True
                true_count += 1
    return true_count


def main():
    args = parser.parse_args()
    cfg = get_cfg_defaults()

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    cfg.merge_from_file(args.config)

    loss_fn = LineSegmentLoss(cfg)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # rank, world_size = setup_distributed(port=args.port)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(rank)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        #logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    #model = DeepLabV3Plus(cfg)
    model = build_model(cfg).cuda()

    if os.path.exists(cfg.train.load_from):
        print('load from: ', cfg.train.load_from)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(cfg.train.load_from, map_location=device), strict=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model.cuda()

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg, 'train_u', args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg, 'train_l', args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg, 'val', is_train=False)

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, cfg.train.batch_size,
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l,
                               collate_fn=get_collate_fn(cfg))
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg.train.batch_size,
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler, collate_fn=get_collate_fn(cfg))

    # mlsd_train_loader = get_train_dataloader(cfg)
    # mlsd_val_loader = get_val_dataloader(cfg)

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
        print(args.prev_best)
        checkpoint = torch.load(os.path.join(args.prev_best, 'best_sAP.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if rank == 0:
            logger.info('****** Load pretrained weights ******')

    model_org = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    for epoch in range(epoch + 1, cfg.train.num_train_epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best fscore: {:.2f}, Previous sAP10: {:.2f}, epoch: '
                        '{:.2f}'.format(epoch, optimizer.param_groups[0]['lr'], previous_best, prev_sAP, best_epoch))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        train_avg_loss = AverageMeter()
        train_avg_center_loss = AverageMeter()
        train_avg_replacement_loss = AverageMeter()
        train_avg_line_seg_loss = AverageMeter()
        train_avg_junc_seg_loss = AverageMeter()
        train_avg_match_loss = AverageMeter()
        train_avg_match_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, (batch,
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            imgs = batch["xs"].cuda()
            label = batch["ys"].cuda()

            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            # if i % 100 == 0:
            #    os.system("nvidia-smi")

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                # conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1)
                # mask_u_w_mix = pred_u_w_mix.argmax(dim=1)
                mask_u_w_mix = pred_u_w_mix

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = imgs.shape[0], img_u_w.shape[0]

            # preds, preds_fp = model(torch.cat((imgs, img_u_w)), True)
            preds = model(torch.cat((imgs, img_u_w)))

            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            # pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w_old = pred_u_w[:, 7:8, :, :].softmax(dim=1).max(dim=1)[0]
            conf_u_w = pred_u_w.softmax(dim=1)
            # mask_u_w = pred_u_w.argmax(dim=1)
            mask_u_w = pred_u_w

            # FULHACK
            # cutmix_box1, cutmix_box2 = cutmix_box1.expand(cfg.train.batch_size, 16, 512, 512), cutmix_box2.expand(cfg.train.batch_size, 16, 512, 512)
            cutmix_box1, cutmix_box2 = cutmix_box1.unsqueeze(1).repeat(1, 16, 1, 1), cutmix_box2.unsqueeze(1).repeat(
                1, 16, 1, 1)
            ignore_mask_mix = ignore_mask_mix.unsqueeze(1).repeat(1, 16, 1, 1)
            ignore_mask = ignore_mask.unsqueeze(1).repeat(1, 16, 1, 1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1[:, :, ::2, ::2] == 1] = mask_u_w_mix[cutmix_box1[:, :, ::2, ::2] == 1]
            conf_u_w_cutmixed1[cutmix_box1[:, :, ::2, ::2] == 1] = conf_u_w_mix[cutmix_box1[:, :, ::2, ::2] == 1]
            ignore_mask_cutmixed1[cutmix_box1[:, :, :, :] == 1] = ignore_mask_mix[cutmix_box1[:, :, :, :] == 1]

            mask_u_w_cutmixed2[cutmix_box2[:, :, ::2, ::2] == 1] = mask_u_w_mix[cutmix_box2[:, :, ::2, ::2] == 1]
            conf_u_w_cutmixed2[cutmix_box2[:, :, ::2, ::2] == 1] = conf_u_w_mix[cutmix_box2[:, :, ::2, ::2] == 1]
            ignore_mask_cutmixed2[cutmix_box2[:, :, :, :] == 1] = ignore_mask_mix[cutmix_box2[:, :, :, :] == 1]


            loss_dict = loss_fn(pred_x, label, batch["gt_lines_tensor_512_list"],
                                batch["sol_lines_512_all_tensor_list"])
            loss_x = loss_dict['loss']

            ignore_mask_cutmixed1 = ignore_mask_cutmixed1[:, :, ::2, ::2]

            loss_u_s1 = abs(criterion_u(pred_u_s1, mask_u_w_cutmixed1))
            # loss_u_s1 = abs(criterion_u(pred_u_s1, pred_u_w))
            loss_u_s1 = loss_u_s1 * (conf_u_w_old >= cfg.unimatch.conf_thresh)
            loss_u_s1 = loss_u_s1.sum() / (16.0 * cfg.train.batch_size * (conf_u_w_old >= cfg.unimatch.conf_thresh).sum().item())
            # loss_u_s1 = loss_u_s1.sum() / 65536.0
            # loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            ignore_mask_cutmixed2 = ignore_mask_cutmixed2[:, :, ::2, ::2]

            loss_u_s2 = abs(criterion_u(pred_u_s2, mask_u_w_cutmixed2))
            # loss_u_s2 = abs(criterion_u(pred_u_s2, pred_u_w))
            loss_u_s2 = loss_u_s2 * (conf_u_w_old >= cfg.unimatch.conf_thresh)
            loss_u_s2 = loss_u_s2.sum() / (16.0 * cfg.train.batch_size * (conf_u_w_old >= cfg.unimatch.conf_thresh).sum().item())
            # loss_u_s2 = loss_u_s2.sum() / 65536.0
            # loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            # loss_u_w_fp = loss_fn(pred_u_w_fp, mask_u_w, batch["gt_lines_tensor_512_list"], batch["sol_lines_512_all_tensor_list"])
            # loss_u_w_fp = loss_u_w_fp['loss']

            # loss = loss_x
            loss = (loss_x * 9.0 + loss_u_s1 * 0.5 + loss_u_s2 * 0.5) / 10.0
            # loss = (loss_x * 19.0 + loss_u_s1 * 0.5 + loss_u_s2 * 0.5) / 20.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            # total_loss_w_fp.update(loss_u_w_fp.item())

            train_avg_loss.update(loss.item(), 1)
            train_avg_center_loss.update(loss_dict['center_loss'].item(), 1)
            train_avg_replacement_loss.update(loss_dict['displacement_loss'].item(), 1)
            train_avg_line_seg_loss.update(loss_dict['line_seg_loss'].item(), 1)
            train_avg_junc_seg_loss.update(loss_dict['junc_seg_loss'].item(), 1)
            train_avg_match_loss.update(float(loss_dict['match_loss']), 1)
            train_avg_match_ratio.update(loss_dict['match_ratio'], 1)

            mask_ratio = ((conf_u_w >= cfg.unimatch.conf_thresh) & (ignore_mask[:, :, ::2, ::2] != 255)).sum().item() / \
                         (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                # writer.add_scalar('train/loss_x', loss_x.item(), iters)
                # writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                # writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if len(trainloader_u) > 0 and (i % max(1, (len(trainloader_u) // 8)) == 0) and (rank == 0):
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                    '{:.3f}, avg = {:.3f}, c: {:.3f}, d: {:.3f}, l: {:.3f}, '
                    'junc:{:.3f}, m:{:.3f}, m_r:{:.2f} '.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                    total_mask_ratio.avg, train_avg_loss.avg, train_avg_center_loss.avg, train_avg_replacement_loss.avg,
                    train_avg_line_seg_loss.avg, train_avg_junc_seg_loss.avg, train_avg_match_loss.avg,
                    train_avg_match_ratio.avg))

        m = evaluate(model, valloader, cfg, epoch, logger, rank, is_wireframe=args.is_wireframe)

        fscore = m['fscore']
        sAP = m["sAP10"]

        is_best = fscore > previous_best
        is_best_sAP = sAP > prev_sAP
        previous_best = max(fscore, previous_best)
        prev_sAP = max(sAP, prev_sAP)
        if rank == 0:
            checkpoint = {
                'model': model_org.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'previous_best': previous_best,
                'previous_sAP': prev_sAP,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            if is_best_sAP:
                best_epoch = epoch
                torch.save(checkpoint, os.path.join(args.save_path, 'best_sAP.pth'))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
