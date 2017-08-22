import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from train_att import DrawModule

batch_size=3
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
    if use_cuda:
        numpy_images=input.cpu().numpy()
    else:
        numpy_images = input.numpy()
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
    model = DrawModule(batch_size=batch_size, enc_size=enc_size, dec_size=dec_size, z_size=z_size, \
                       use_att=use_att, T=T,img_size=img_size,N=N).cuda()
else:
    model = DrawModule(batch_size=batch_size, enc_size=enc_size, dec_size=dec_size, z_size=z_size, use_att=use_att, T=T)
model.load_state_dict(torch.load('../models/DRAW/DRAW5'))
imgs=model.generate(batch_size)
print type(imgs)
print len(imgs[0]),len(imgs),type(imgs[0][0]),imgs[0][0].shape

h_dec_test=Variable(torch.randn(batch_size,dec_size)).cuda()
gx,gy,delta,gamma,var=model.attention_param(h_dec_test)
print gx,gy,delta,gamma,var
Fx,Fy=model.filterbank(gx,gy,delta,var)

for i in range(T):
    img_last=imgs[i]
    # print type(img_last),img_last.shape

    tensor_img_last=torch.FloatTensor(img_last)
    tensor_img_reshape=tensor_img_last.view(-1,1,28,28)

    img_grid=torchvision.utils.make_grid(tensor_img_reshape)
    plt.figure(i)
    imshow(img_grid)

train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_dataset = torchvision.datasets.MNIST(root='../data', download=True, train=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4)

train_iter=iter(train_loader)
images,labels=train_iter.next()
x=Variable(images).cuda() if use_cuda else Variable(images)
print x.data.size()
model.batch_size=batch_size
mus,sigmas,logsigmas,x_recons=model.forward(x)
# loss_test=loss_func(x,mus,sigmas,logsigmas,x_recons)
x_recons_tensors=x_recons.data
grid_recons=torchvision.utils.make_grid(x_recons_tensors)
imshow(grid_recons)
