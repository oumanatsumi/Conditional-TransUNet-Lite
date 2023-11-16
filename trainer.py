import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from medpy import metric
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from utils import FocalLoss
import feature_utils.img_feature_extractor as ife
from torchvision import transforms
import numpy as np


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    dataset = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of dataset is: {}".format(len(dataset)))
    train_size = int(len(dataset) * 0.8)
    valid_size = int((len(dataset) - train_size)*0.5)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    alpha = 0.5
    beta = 0.9
    wce_weight = torch.from_numpy(np.array([200000/700, 1])).float().cuda()
    ce_loss = CrossEntropyLoss()
    wce_loss = CrossEntropyLoss(wce_weight)
    dice_loss = DiceLoss(num_classes)
    focal_loss = FocalLoss(weight=wce_weight)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # 得到image_batch特征，加入到模型中
            mats = ife.tensor2cv_mat(image_batch)
            batch_embedded_feature = ife.batch_embedding(mats, 170)
            model.transformer.embeddings.feature_embeddings = nn.Parameter(torch.Tensor(batch_embedded_feature).cuda())
            # print(torch.unique(label_batch, return_counts=True))
            outputs = model(image_batch)
            # loss_itm = ce_loss(ITM_logits, ITM_labels)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_wce = wce_loss(outputs, label_batch[:].long())
            loss_focal = focal_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = alpha * loss_focal + (1-alpha) * loss_dice
            # loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/train_total_loss', loss, iter_num)
            writer.add_scalar('info/train_loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/train_loss_wce', loss_wce, iter_num)
            writer.add_scalar('info/train_loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/train_loss_dice', loss_dice, iter_num)


            # logging.info('iteration %d : train_loss : %f, train_loss_ce: %f, train_loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 200 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # valid test
            valid_cnt = 0
            hd95_cnt = 0
            valid_loss = 0.0
            valid_loss_ce = 0.0
            valid_loss_wce = 0.0
            valid_loss_focal = 0.0
            valid_loss_dice = 0.0
            valid_mean_hd95 = 0.0
            for i_valid_batch, sampled_valid_batch in enumerate(validloader):
                valid_cnt = valid_cnt + 1
                valid_image_batch, valid_label_batch = sampled_valid_batch['image'], sampled_valid_batch['label']
                valid_image_batch, valid_label_batch = valid_image_batch.cuda(), valid_label_batch.cuda()
                mats = ife.tensor2cv_mat(valid_image_batch)
                batch_embedded_feature = ife.batch_embedding(mats, 170)
                model.transformer.embeddings.feature_embeddings = nn.Parameter(torch.Tensor(batch_embedded_feature).cuda())
                outputs = model(valid_image_batch)
                valid_loss_ce += ce_loss(outputs, valid_label_batch[:].long()).item()
                valid_loss_wce += wce_loss(outputs, valid_label_batch[:].long()).item()
                valid_loss_focal += focal_loss(outputs, valid_label_batch[:].long()).item()
                valid_loss_dice += dice_loss(outputs, valid_label_batch, softmax=True).item()
                valid_loss += alpha * valid_loss_focal + (1-alpha) * valid_loss_dice
                # valid_loss = valid_loss_ce
                ot = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=False).cpu().detach().numpy()
                vl = valid_label_batch.cpu().detach().numpy()
                if ot.any() and vl.any():
                    valid_mean_hd95 += metric.binary.hd95(ot, vl)
                    hd95_cnt = hd95_cnt + 1
            valid_loss /= valid_cnt
            valid_loss_ce /= valid_cnt
            valid_loss_wce /= valid_cnt
            valid_loss_focal /= valid_cnt
            valid_loss_dice /= valid_cnt
            if hd95_cnt == 0:
                valid_mean_hd95 = float('inf')
            else:
                valid_mean_hd95 /= hd95_cnt
            writer.add_scalar('info/valid_total_loss', valid_loss, iter_num)
            writer.add_scalar('info/valid_loss_ce', valid_loss_ce, iter_num)
            writer.add_scalar('info/valid_loss_wce', valid_loss_wce, iter_num)
            writer.add_scalar('info/valid_loss_focal', valid_loss_focal, iter_num)
            writer.add_scalar('info/valid_loss_dice', valid_loss_dice, iter_num)
            logging.info('iteration %d : train_loss : %f, train_loss_ce: %f, train_loss_wce: %f,train_loss_focal: %f, train_loss_dice: %f '
                         'valid_loss : %f, valid_loss_ce: %f, valid_loss_wce : %f, valid_loss_focal: %f, valid_loss_dice: %f, valid_mean_hd95: %f'
                         % (iter_num, loss.item(), loss_ce.item(),loss_wce.item(),loss_focal.item(), loss_dice.item(),
                            valid_loss, valid_loss_ce, valid_loss_wce, valid_loss_focal, valid_loss_dice, valid_mean_hd95))

        save_interval = 40  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        # if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            # test
            test_cnt = 0
            hd95_cnt = 0
            test_loss = 0.0
            test_loss_ce = 0.0
            test_loss_wce = 0.0
            test_loss_focal = 0.0
            test_loss_dice = 0.0
            test_mean_hd95 = 0.0
            for i_test_batch, sampled_test_batch in enumerate(testloader):
                test_cnt = test_cnt + 1
                test_image_batch, test_label_batch = sampled_test_batch['image'], sampled_test_batch['label']
                test_image_batch, test_label_batch = test_image_batch.cuda(), test_label_batch.cuda()
                mats = ife.tensor2cv_mat(test_image_batch)
                batch_embedded_feature = ife.batch_embedding(mats, 170)
                model.transformer.embeddings.feature_embeddings = nn.Parameter(
                    torch.Tensor(batch_embedded_feature).cuda())
                outputs = model(test_image_batch)
                test_loss_ce += ce_loss(outputs, test_label_batch[:].long()).item()
                test_loss_wce += wce_loss(outputs, test_label_batch[:].long()).item()
                test_loss_focal += focal_loss(outputs, test_label_batch[:].long()).item()
                test_loss_dice += dice_loss(outputs, test_label_batch, softmax=True).item()
                test_loss += alpha * test_loss_focal + (1-alpha) * test_loss_dice
                ot = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=False).cpu().detach().numpy()
                vl = test_label_batch.cpu().detach().numpy()
                if ot.any() and vl.any():
                    test_mean_hd95 += metric.binary.hd95(ot, vl)
                    hd95_cnt = hd95_cnt + 1
            test_loss /= test_cnt
            test_loss_ce /= test_cnt
            test_loss_wce /= test_cnt
            test_loss_focal /= test_cnt
            test_loss_dice /= test_cnt
            if hd95_cnt == 0:
                test_mean_hd95 = float('inf')
            else:
                test_mean_hd95 /= test_cnt
            writer.add_scalar('info/test_total_loss', test_loss, iter_num)
            writer.add_scalar('info/test_loss_ce', test_loss_ce, iter_num)
            writer.add_scalar('info/test_loss_wce', test_loss_wce, iter_num)
            writer.add_scalar('info/test_loss_focal', test_loss_dice, iter_num)
            writer.add_scalar('info/test_loss_dice', test_loss_dice, iter_num)

            logging.info(
                'TEST RESULT : test_loss : %f,  test_loss_ce: %f, test_loss_wce: %f,test_loss_focal: %f ,test_loss_dice: %f'
                % (test_loss, test_loss_ce,test_loss_wce, test_loss_focal, test_loss_dice))
            logging.info(
                'DSC :  %f, hd95: %f,'
                % (1-test_loss_dice, test_mean_hd95))

            iterator.close()
            break

    writer.close()
    return "Training Finished!"