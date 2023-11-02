import torch
import torchvision
import netron
import img_feature_extractor as ife
import cv2
from PIL import Image
import numpy as np
import math

# dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
# model = torch.load("..\\model\\TU_Synapse224\\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs12_224\\epoch_14.pth")
#
# input_names = ["input_1"]
# output_names = ["output_1"]
#
# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=input_names, output_names=output_names)

# img=cv2.imread("D:\\learn\\python\\centerPrediction\\data\\MPR(6).jpg")
#
# feature = ife.extract_feature(img, (512, 512), 170)
# embedding = torch.nn.Embedding(512,14)
# e1 = embedding(torch.LongTensor(feature))
# print(e1)

cnt = 0
pathLoc ="D:\\learn\\python\\TransUNetProject\\data\\Synapse\\train_npz\\"
for i in range(195):
    path = pathLoc + "case0040_slice" + str(i).zfill(3) + ".npz"
    data = np.load(path)
    image = data["image"]
    label = data["label"]
    # print("image:  \n")
    # print(image)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            print(image[x][y])
            # if(not math.isclose(image[x][y],0.0,rel_tol = 0.000000001)):
            #     cnt = cnt + 1
    # print("label:  \n")
    # print(label)
    # im = Image.fromarray(image)
    # im.show()
print(cnt)
