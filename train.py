import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

batch_size=8
enc_size=256
dec_size=256
N=5
img_size=28
T=5
use_att=False

train_transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_dataset=torchvision.datasets.MNIST(root='./data',download=True,train=True,transform=train_transforms)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)

#how many batch=num of images/batch_size
print "train dataset length:",len(train_loader)
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

train_iter=iter(train_loader)
images,labels=train_iter.next()
#return [channels,height,widthxbatch_size] (3L,32L,242L)
images_grid=torchvision.utils.make_grid(images)
# imshow(images_grid)



class DrawModule(nn.Module):
    def __init__(self,batch_size,enc_size,use_att,T):
        super(DrawModule, self).__init__()
        self.use_att=use_att
        self.batch_size=batch_size
        self.enc_size=enc_size
        self.T=T

        if use_att:
            self.encoder=nn.LSTMCell(2*N*N,enc_size)
        else:
            self.encoder=nn.LSTMCell(784,enc_size)
        return

    def Encode(self,input_x,(h_enc_prev,enc_state)):

        return self.encoder(input_x,(h_enc_prev,enc_state))

    def forward(self,input):
        assert(input.size()[2]==input.size()[3])
        x=input.view(-1,input.size()[2]*input.size()[3])
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))
        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))

        # for i in range(self.T):
        #     input_x=self.read(x)
        #     h_enc_prev,enc_state=self.Encode(input_x,(h_enc_prev,enc_state))

        h_enc_prev, enc_state = self.encoder(x, (h_enc_prev, enc_state))

        return

    # def read(self,x,x_hat,h_dec_prev):
    #     if self.use_att:
    #         return torch.cat((x,x_hat),1)
    #     else:
    #         return torch.cat((x,x_hat),1)
    def read(self,x):
        return x

model=DrawModule(batch_size=batch_size,enc_size=enc_size,use_att=use_att,T=T)
model.forward(images)