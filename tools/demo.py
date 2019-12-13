# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

from dataset import img_coco
from core.inference import get_final_preds
import dataset
import models
import cv2
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

color1 = [(179,0,0),(228,26,28),(255,255,51),
    (49,163,84), (0,109,45), (255,255,51),
    (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
    (217,95,14), (254,153,41),(255,255,51),
    (44,127,184),(0,0,255)]

link_pairs1 = [
        [15, 13], [13, 11], [11, 5], 
        [12, 14], [14, 16], [12, 6], 
        [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
        [9, 7], [7,5], [5, 6],
        [6, 8], [8, 10],
        ]

point_color1 = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,51),(255,255,51),
            (254,153,41),(44,127,184),
            (217,95,14),(0,0,255),
            (255,255,51),(255,255,51),(228,26,28),
            (49,163,84),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

def draw_kpts(img,all_bbs,all_pts, all_scales):
    #print(all_bbs)
    #print(all_pts)
    #print(all_scales)
    for box, pts, scale in zip(all_bbs, all_pts, all_scales):
        #print(box)
        #print(pts)
        #print(scale)
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]), (0,255,0),2)
        pts = pts / scale
        x1,y1 = box[:2]
        pts[:,0] += x1
        pts[:,1] += y1
        pts.astype(np.int32)
        for limb in link_pairs1:
            ps_id = limb[0]
            pe_id = limb[1]
            x1 = pts[ps_id][0]
            y1 = pts[ps_id][1]
            x2 = pts[pe_id][0]
            y2 = pts[pe_id][1]
            cv2.line(img, (x1,y1), (x2,y2), color1[ps_id],2)
    #cv2.imshow('kpts', img)
    #cv2.waitKey()
    return img
 
def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    model.eval()
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    print(cfg.DATASET.DATASET)
    print(cfg.DATASET.ROOT)
    print(cfg.DATASET.TEST_SET)
    img_sets = img_coco.IMGCOCO(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,transforms.Compose([transforms.ToTensor(), normalize,]))
    all_imgids = img_sets.image_set
    with torch.no_grad():
        for idx, imid in enumerate(all_imgids):
            #if idx >= 20:
            #    break
            persons, all_bbs, all_scales, ori_img, imname = img_sets.generate_pose_input(imid)
            all_pts = []
            for pid, person in enumerate(persons):
                outputs = model(person)
                #print(outputs.numpy().shape)
                preds, maxvals = get_final_preds(cfg, outputs.clone().cpu().numpy(), [],[])
                kpts = preds[0,:] * 4
                all_pts.append(kpts)
                #print(kpts)
                #print(kpts.astype(np.int32))
                #draw_kpts(ori_persons[pid], kpts)
                #cv2.imshow('people', person)
                #cv2.waitKey()
            vis_img = draw_kpts(ori_img,all_bbs, all_pts, all_scales)
            out_path = os.path.join('results', imname)
            cv2.imwrite(out_path, vis_img)
            
    
    return
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([transforms.ToTensor(),normalize,]))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    return
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
