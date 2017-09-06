import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

##Two remaining question~


# why doesn't it work?
# train_transforms=transforms.Compose([transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),transforms.ToTensor()])
train_transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_dataset=torchvision.datasets.MNIST(root='./data',download=True,train=True,transform=train_transforms)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=8,shuffle=True,num_workers=4)

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
images=torchvision.utils.make_grid(images)
imshow(images)