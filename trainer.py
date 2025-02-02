import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
import feature_utils.img_feature_extractor as ife
from torchvision import transforms


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
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
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
            outputs, ITM_labels, ITM_logits = model(image_batch)
            loss_itm = ce_loss(ITM_logits, ITM_labels)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
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
            writer.add_scalar('info/train_loss_dice', loss_dice, iter_num)

            # logging.info('iteration %d : train_loss : %f, train_loss_ce: %f, train_loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # valid test
            valid_cnt = 0
            test_loss = 0.0
            test_loss_ce = 0.0
            test_loss_dice = 0.0
            for i_valid_batch, sampled_valid_batch in enumerate(validloader):
                valid_cnt = valid_cnt + 1
                valid_image_batch, valid_label_batch = sampled_valid_batch['image'], sampled_valid_batch['label']
                valid_image_batch, valid_label_batch = valid_image_batch.cuda(), valid_label_batch.cuda()
                mats = ife.tensor2cv_mat(valid_image_batch)
                batch_embedded_feature = ife.batch_embedding(mats, 170)
                model.transformer.embeddings.feature_embeddings = nn.Parameter(torch.Tensor(batch_embedded_feature).cuda())
                outputs, ITM_labels, ITM_logits = model(valid_image_batch)
                test_loss_ce += ce_loss(outputs, valid_label_batch[:].long()).item()
                test_loss_dice += dice_loss(outputs, valid_label_batch, softmax=True).item()
                test_loss += 0.5 * loss_ce + 0.5 * loss_dice.item()
            test_loss /= valid_cnt
            test_loss_ce /= valid_cnt
            test_loss_dice /= valid_cnt
            writer.add_scalar('info/test_total_loss', test_loss, iter_num)
            writer.add_scalar('info/test_loss_ce', test_loss_ce, iter_num)
            writer.add_scalar('info/test_loss_dice', test_loss_dice, iter_num)
            logging.info('iteration %d : train_loss : %f, train_loss_ce: %f, train_loss_dice: %f train_loss : %f, train_loss_ce: %f, train_loss_dice: %f'
                         % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), test_loss, test_loss_ce, test_loss_dice))

        save_interval = 20  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"