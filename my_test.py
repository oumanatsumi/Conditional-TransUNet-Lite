import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from medpy import metric
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import *
from torch.utils.data import DataLoader
# from networks.vit_seg_modeling import VisionTransformer as ViT_sceg
# from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
import netron
from torch import onnx as onnx
from tensorboardX import SummaryWriter as SummaryWriter
from torchinfo import summary
from torchvision import transforms
from utils import DiceLoss
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import feature_utils.img_feature_extractor as ife


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/TAVR/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='TAVR', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/list_TAVR', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=231116, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
# summary(net, input_size=(1,3,224,224))

model_path = 'D:\learn\python\TransUNet_LiteProject\model\TU_TAVR224\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs4_224'
model_path += '\\epoch_149.pth'
loaded = torch.load(model_path)
del loaded['transformer.embeddings.feature_embeddings']
net.load_state_dict(loaded, strict=False)
model = net
model.eval()
base_lr = args.base_lr
num_classes = args.num_classes
batch_size = args.batch_size * args.n_gpu
# max_iterations = args.max_iterations
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
dataset = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                          transform=transforms.Compose(
                              [RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of dataset is: {}".format(len(dataset)))
train_size = int(len(dataset) * 0.8)
valid_size = int((len(dataset) - train_size) * 0.5)
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

alpha = 0.5
beta = 0.9
# wce_weight = torch.from_numpy(np.array([200000/700, 1])).float().cuda()
ce_loss = CrossEntropyLoss()
# wce_loss = CrossEntropyLoss(wce_weight)
dice_loss = DiceLoss(num_classes)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
iter_num = 0
max_epoch = args.max_epochs
max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
best_performance = 0.0

# test
test_cnt = 0
hd95_cnt = 0
test_loss = 0.0
test_loss_ce = 0.0
test_loss_dice = 0.0
test_mean_hd95 = 0.0
for i_test_batch, sampled_test_batch in enumerate(testloader):
    test_cnt = test_cnt + 1
    test_image_batch, test_label_batch = sampled_test_batch['image'], sampled_test_batch['label']
    test_image_batch, test_label_batch = test_image_batch.cuda(), test_label_batch.cuda()
    # 得到image_batch特征，加入到模型中
    mats = ife.tensor2cv_mat(test_image_batch)
    batch_embedded_feature = ife.batch_embedding(mats, 170)
    model.transformer.embeddings.feature_embeddings = nn.Parameter(torch.Tensor(batch_embedded_feature).cuda())
    outputs = model(test_image_batch)
    test_loss_ce += ce_loss(outputs, test_label_batch[:].long()).item()
    test_loss_dice += dice_loss(outputs, test_label_batch, softmax=True).item()
    test_loss += alpha * test_loss_ce + (1 - alpha) * test_loss_dice
    ot = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=False).cpu().detach().numpy()
    vl = test_label_batch.cpu().detach().numpy()
    if ot.any() and vl.any():
        test_mean_hd95 += metric.binary.hd95(ot, vl)
        hd95_cnt = hd95_cnt + 1
test_loss /= test_cnt
test_loss_ce /= test_cnt
test_loss_dice /= test_cnt
if hd95_cnt == 0:
    test_mean_hd95 = float('inf')
else:
    test_mean_hd95 /= test_cnt
logging.info(
    'TEST RESULT : test_loss : %f, test_loss_ce: %f, test_loss_dice: %f'
    % (test_loss, test_loss_ce, test_loss_dice))
logging.info(
    'DSC :  %f, hd95: %f,'
    % (1 - test_loss_dice, test_mean_hd95))

print('DSC :  %f, hd95: %f'
    % (1 - test_loss_dice, test_mean_hd95))

# summary(net,(1,1,224,224))