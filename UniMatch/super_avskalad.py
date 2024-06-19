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

from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule
from pathlib import Path


parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--save_images', default='/out', type=str)
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


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")


def main():
    args = parser.parse_args()
    cfg = get_cfg_defaults()

    output_dir = Path(TRAINING_PATH, args.save_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.config.endswith('\r'):
        args.config = args.config[:-1]
    cfg.merge_from_file(args.config)

    loss_fn = LineSegmentLoss(cfg)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = build_model(cfg).cuda()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    trainset = SemiDataset(cfg, 'train_l', args.labeled_id_path)

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg.train.batch_size,
                             pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler,
                             collate_fn=get_collate_fn(cfg))

    previous_best = 0.0
    prev_sAP = 0.0
    epoch = -1
    best_epoch = 0

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    prof.__enter__()

    for epoch in range(epoch + 1, cfg.train.num_train_epochs):

        model.train()
        trainsampler.set_epoch(epoch)
        print("Epoch: ", epoch)

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

        prof.step()


if __name__ == '__main__':
    main()
