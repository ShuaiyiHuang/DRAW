import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from train import DrawModule

batch_size=8
enc_size=256
dec_size=256
z_size=10
N=5
img_size=28
T=5
use_att=False
use_cuda=True
lrt=1e-2
log_interval=150


def imshow(input):
    numpy_images=input.numpy()
    ##what is std=[a,b,c] and mean=[e,f,g]??
    std=0.5
    mean=0.5
    img=numpy_images*std+mean
    img_transpose=np.transpose(img,(1,2,0))
    #matplotlib use [height,width,channels],where channels is R,G,B
    #images like Cifar,MNIST...from pytorch is of [batchsize,channels,height,width],where channels is R,G,B.
    plt.imshow(img_transpose)
    #add this to show in pycharm
    plt.show()
    return

if use_cuda:
    model = DrawModule(batch_size=batch_size, enc_size=enc_size, dec_size=dec_size, z_size=z_size, use_att=use_att, T=T).cuda()
else:
    model = DrawModule(batch_size=batch_size, enc_size=enc_size, dec_size=dec_size, z_size=z_size, use_att=use_att, T=T)
model.load_state_dict(torch.load('../models/DRAW/DRAW9'))
imgs=model.generate(4)
print type(imgs)
print len(imgs[0]),len(imgs),type(imgs[0][0]),imgs[0][0].shape

for i in range(T):
    img_last=imgs[i]
    print type(img_last),img_last.shape

    tensor_img_last=torch.FloatTensor(img_last)
    tensor_img_reshape=tensor_img_last.view(-1,1,28,28)

    img_grid=torchvision.utils.make_grid(tensor_img_reshape)
    plt.figure(i)
    imshow(img_grid)

# images_grid=torchvision.utils.make_grid(images)