'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, coco_caption_eval


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)

        loss = model(image, caption)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header):
        image = image.to(device)
        captions = model.generate(image, sample=True, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})

    return result


def main(args, config):
    # 初始化分布式模式，args包含分布式训练相关配置的参数对象
    utils.init_distributed_mode(args)
    # 根据传入的参数 args.device 设置 PyTorch 的设备对象(三种取值：cpu, cuda/cuda:0, cuda:1/cuda:2)
    device = torch.device(args.device)

    # fix the seed for reproducibility(固定随机数种子，以确保代码运行结果的可重复性)
    seed = args.seed + utils.get_rank()  # 设置的固定种子加上当前进程的排名
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数生成器的种子值为 seed
    np.random.seed(seed)  # 设置随机数种子,与manual随机数序列一致
    random.seed(seed)  # 与上面功能相同
    cudnn.benchmark = True  # 提高运行速度，但是在某些情况下可能会带来一些额外的开销

    #### Dataset #### 
    print("Creating captioning dataset")

    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)
    # 通过config里的设置创建名为caption_coco的训练集、验证集、测试集
    # 检查是否采用分布式训练，采用则获取当前总任务数和全局排名，然后调用 create_sampler 函数创建数据采样器。
    # 如果未启用分布式训练，则直接将 samplers 设为三个 None 值。
    if args.distributed:
        num_tasks = utils.get_world_size()  # 获取分布式中任务总数
        global_rank = utils.get_rank()  # 获取当前进程全局排名
        samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks,
                                  global_rank)  # 创建数据采样器
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size']] * 3, num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                         vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                         prompt=config['prompt'])  # 创建blip解码器

    model = model.to(device)  # 将模型移动到device设备上

    model_without_ddp = model  # 另外将model赋值给不使用分布式的模型model_without_ddp
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # 将模型包装在 torch.nn.parallel.DistributedDataParallel 中
        model_without_ddp = model.module  # 保存未经包装的原始模型
    # 创建AdamW 优化器对象,并传入了模型的参数、学习率以及权重衰减参数
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)  # 在训练数据加载器的采样器中设置一个新的 epoch

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            # 根据余弦退火的策略动态地调整学习率
            train_stats = train(model, train_loader, optimizer, epoch, device)
            # 训练模型并返回训练过程中的统计信息

        val_result = evaluate(model_without_ddp, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d' % epoch, remove_duplicate='image_id')
        print("val_result_file:", val_result_file)
        test_result = evaluate(model_without_ddp, test_loader, device, config)
        test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d' % epoch, remove_duplicate='image_id')
        print("test_result_file", test_result_file)
        # 对主进程进行模型保存、评估结果保存操作
        if utils.is_main_process():
            coco_val = coco_caption_eval(config['coco_gt_root'], val_result_file, 'val')
            coco_test = coco_caption_eval(config['coco_gt_root'], test_result_file, 'test')
            # 计算评估指标，保存到log_stats中
            if args.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             }
                with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                # 比较验证集评估结果，保存最佳性能的模型
                if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                    best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                    best_epoch = epoch
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             **{f'test_{k}': v for k, v in coco_test.eval.items()},
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                             }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/power-self.yaml')
    parser.add_argument('--output_dir', default='output/Caption_coco')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
