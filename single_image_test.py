import argparse
import datetime
import logging
import os
import random

import cv2
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchinfo import summary
import feature_utils.img_feature_extractor as ife

image_name_list = ['000020.npz', '000044.npz','000072.npz', '000073.npz', '000182.npz', '000184.npz','000197.npz','000224.npz','000255.npz']

model_path = 'C:\\Users\\Administrator\\OneDrive\\毕设\\实验数据\\model\\'
model_path += 'best_epoch_149.pth'

output_path = './test_result/chap4/'
output_name = '/best_epoch_149.jpg'

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
                    default=24, help='batch_size per gpu')
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

loaded = torch.load(model_path)
del loaded['transformer.embeddings.feature_embeddings']
net.load_state_dict(loaded, strict=False)
model = net
model.eval()

for image_name in image_name_list:
    image_path = '../data/TAVR/test/chap4/' + image_name

    image = np.load(image_path)['image']
    image = cv2.resize(image, (224, 224))
    # cv2.imshow("Src", image)
    cv2.imwrite(output_path+ image_name[:6]+"/src.jpg", image*255.0)
    label = np.load(image_path)['label']
    label = cv2.resize(label, (224, 224))
    cv2.imwrite(output_path+ image_name[:6]+"/GT.jpg", label*255.0)
    image = np.reshape(image, (1, 1, 224, 224))
    inputs = torch.tensor(image).cuda()
    # 得到image_batch特征，加入到模型中
    mats = ife.tensor2cv_mat(inputs)
    batch_embedded_feature = ife.batch_embedding(mats, 170)
    model.transformer.embeddings.feature_embeddings = nn.Parameter(torch.Tensor(batch_embedded_feature).cuda())
    outputs = net(inputs)
    outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
    outputs = np.reshape(outputs.cpu().detach().numpy(), (224, 224))
    # cv2.imshow("Prediction", outputs*255.0)
    cv2.imwrite(output_path + image_name[:6] + output_name, outputs * 255.0)
    print(output_path + image_name[:6] + output_name+ " saved")


