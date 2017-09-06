import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from utility import save_image,xrecons_grid
import time

batch_size=4
enc_size=256
dec_size=256
z_size=10
N=5
img_size=28
A=img_size
B=img_size
T=10
use_att=True
use_cuda=True
lrt=1e-3
log_interval=150
epoch_num=20

#you shouldn't use normalize if you use BCE loss and sigmoid for x_recons
# train_transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_transforms=transforms.Compose([transforms.ToTensor()])
train_dataset=torchvision.datasets.MNIST(root='../data',download=True,train=True,transform=train_transforms)
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)

test_transforms=transforms.Compose([transforms.ToTensor()])
test_dataset=torchvision.datasets.MNIST(root='../data',download=True,train=False,transform=test_transforms)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=4,drop_last=True)

#how many batch=num of images/batch_size
print "train dataloader length:",len(train_loader)
print 'test dataloader length:',len(test_loader)
#with mean and std
def imshow(input):
    numpy_images=input.numpy()
    ##what if std=[a,b,c] and mean=[e,f,g]??
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
    def __init__(self,batch_size,enc_size,dec_size,z_size,use_att,T,img_size,N):
        super(DrawModule, self).__init__()
        self.use_att=use_att
        self.batch_size=batch_size
        # print 'batch_size:',self.batch_size
        self.enc_size=enc_size
        self.dec_size=dec_size
        self.T=T
        self.N=N
        self.z_size=z_size
        self.logsigmas, self.sigmas, self.mus= [0] * T, [0] * T, [0] * T
        self.mu_linear=nn.Linear(enc_size,z_size)
        self.sigma_linear=nn.Linear(enc_size,z_size)

        self.attparam_linear=nn.Linear(dec_size,5)
        #list.every element can be of different type
        self.Cs= [0] * T
        self.x_recons=None
        #to modify
        self.A=img_size
        self.B=img_size
        # sigmoid() takes exactly 1 argument (0 given)
        # self.sigmoid=F.sigmoid()
        #correct return class
        self.sigmoid=nn.Sigmoid()

        #if use_cuda,specify lstm.cuda()?
        if self.use_att:
            print 'use attention'
            self.encoder=nn.LSTMCell(2*N*N,enc_size).cuda() if use_cuda else nn.LSTMCell(2*N*N,enc_size)
            self.write_linear=nn.Linear(dec_size,self.N*self.N)
        else:
            print 'no use attention'
            self.encoder=nn.LSTMCell(2*self.A*self.B,enc_size).cuda() if use_cuda else nn.LSTMCell(2*self.A*self.B,enc_size)
            self.write_linear = nn.Linear(dec_size, self.A * self.B)

        self.decoder=nn.LSTMCell(z_size,dec_size).cuda() if use_cuda else nn.LSTMCell(z_size,dec_size)
        return

    def Encode(self,input_x,(h_enc_prev,enc_state)):
        return self.encoder(input_x,(h_enc_prev,enc_state))

    def Decode(self,input_x,(h_dec_prev,dec_state)):
        return self.decoder(input_x,(h_dec_prev,dec_state))

    def forward(self,input):
        # assert(input.size()[2]==input.size()[3])
        x=input.view(-1,input.size()[2]*input.size()[3])
        if use_cuda:
            h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size)).cuda()
            h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size)).cuda()
            enc_state = Variable(torch.zeros(self.batch_size,self.enc_size)).cuda()
            dec_state = Variable(torch.zeros(self.batch_size,self.dec_size)).cuda()
        else:
            h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))
            h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size))
            enc_state = Variable(torch.zeros(self.batch_size, self.enc_size))
            dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        # h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))
        # h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size))
        # enc_state = Variable(torch.zeros(self.batch_size, self.enc_size))
        # dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))

        for i in range(self.T):
            if use_cuda:
                c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)).cuda() if i == 0 else self.Cs[i - 1]
            else:
                c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if i == 0 else self.Cs[i - 1]
            # print x.size(), c_prev.size()
            x_hat=x-self.sigmoid(c_prev)
            input_x=self.read(x,x_hat,h_dec_prev)
            h_enc_prev,enc_state=self.Encode(input_x,(h_enc_prev,enc_state))
            z, self.mus[i], self.logsigmas[i], self.sigmas[i]=self.sampleZ(h_enc_prev)
            h_dec_prev,dec_state=self.Decode(z,(h_dec_prev,dec_state))
            self.Cs[i]= c_prev + self.write(h_dec_prev)

        self.x_recons=self.sigmoid(self.Cs[-1]).view(-1,1,self.A,self.B)
        return self.mus,self.sigmas,self.logsigmas,self.x_recons

    def read(self,x,x_hat,h_dec_prev):
        if self.use_att:
            x=x.view(-1,self.B,self.A)
            x_hat=x_hat.view(-1,self.B,self.A)
            gx,gy,gamma,delta,var=self.attention_param(h_dec_prev)
            Fx,Fy=self.filterbank(gx,gy,delta,var)
            # print Fy.size(),x.size(),x_hat.size()
            a=torch.matmul(Fy,x)
            #!!!!!
            Fx_t=torch.transpose(Fx,1,2)
            x_att=torch.matmul(a,Fx_t)
            x_att_hat=torch.matmul(torch.matmul(Fy,x_hat),torch.transpose(Fx,1,2))
            x_att_view=x_att.view(-1,self.N*self.N)
            x_att_hat_view=x_att_hat.view(-1,self.N*self.N)
            concat=torch.cat((x_att_view,x_att_hat_view),1)

            x_read=gamma*concat
            # print x_read.size()
            return x_read
        else:
            return torch.cat((x,x_hat),1)
    def attention_param(self,h_dec):
        att_param=self.attparam_linear(h_dec)

        #Tensor:(Tensor) tensor to split.
        #split_size:(int) size of a single chunk.
        #dim:(int) dimension along which to split the tensor.
        g_x_hat,g_y_hat,logvar,log_delta_hat,log_gamma=torch.split(att_param,1,1)
        g_x=0.5*(self.A+1)*(g_x_hat+1)
        g_y=0.5*(self.B+1)*(g_y_hat+1)
        delta=torch.exp(log_delta_hat)*(np.max((self.A,self.B),0)-1)/(self.N-1)
        gamma=torch.exp(log_gamma)
        var=torch.exp(logvar)
        return g_x,g_y,delta,gamma,var

    def filterbank(self,gx,gy,delta,var):
        if use_cuda:
            Fx=Variable(torch.zeros(self.batch_size,self.N,self.A)).cuda()
            Fy=Variable(torch.zeros(self.batch_size,self.N,self.B)).cuda()
        for i in range(self.N):
            mu_i_x = gx + (i - self.N * 0.5 - 0.5) * delta
            for a in range(self.A):
                Fx[:,i,a]=torch.exp(-(a-mu_i_x)*(a-mu_i_x)/(2*var))
        for j in range(self.N):
            mu_j_y=gy+(j-self.N*0.5-0.5)*delta
            for b in range(self.B):
                Fy[:,j,b]=torch.exp(-(b-mu_j_y)*(b-mu_j_y)/(2*var))
        Zx=torch.unsqueeze(torch.unsqueeze(Fx.sum(2).sum(1),1),2).expand_as(Fx)
        Fx=Fx/Zx
        Zy=torch.unsqueeze(torch.unsqueeze(Fy.sum(2).sum(1),1),2).expand_as(Fy)
        Fy=Fy/Zy

        # for batch_id in range(self.batch_size):
        #     Zx=torch.sum(Fx[batch_id,:,:])
        #     #Remember to use .clone,or you will encounter in-place error
        #     Fx[batch_id,:,:]=Fx[batch_id,:,:].clone()/Zx
        # for batch_id in range(self.batch_size):
        #     Zy=torch.sum(Fy[batch_id,:,:])
        #     Fy[batch_id,:,:]=Fy[batch_id,:,:].clone()/Zy
        # print Fx.size(),Fy.size()
        return Fx,Fy

    def write(self,h_dec):
        if self.use_att:
            wt=self.write_linear(h_dec)
            wt=wt.view(-1,self.N,self.N)
            gx,gy,gamma,delta,var=self.attention_param(h_dec)
            Fx,Fy=self.filterbank(gx,gy,delta,var)


            Fy_t=torch.transpose(Fy,1,2)

            a=torch.matmul(Fy_t,wt)

            to_write=torch.matmul(a,Fx)
            # print Fy_t.size(),wt.size(),Fx.size(),a.size(),to_write.size()
            # print 'gamma size:',gamma.size()
            to_write=(1/gamma)*to_write.view(-1,self.A*self.B)
            to_write=to_write.view(-1,self.B,self.A)
        else:
            to_write = self.write_linear(h_dec)
        return to_write

    def normalSample(self):
        return Variable(torch.randn(self.batch_size,self.z_size)).cuda() if use_cuda else Variable(torch.randn(self.batch_size,self.z_size))

    def sampleZ(self,h_enc):
        eps=self.normalSample()
        mu=self.mu_linear(h_enc)    #equation (1)
        logsigma=self.sigma_linear(h_enc) #equation (2)
        sigma=torch.exp(logsigma)
        z=mu+sigma*eps
        return z,mu,logsigma,sigma

    def generate(self,batch_size=64):
        #make sure that input batch_szie equals batch_size of the model
        self.batch_size = batch_size
        if use_cuda:
            h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size), volatile=True).cuda()
            dec_state = Variable(torch.zeros(self.batch_size, self.dec_size), volatile=True).cuda()
        else:
            h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size), volatile=True)
            dec_state = Variable(torch.zeros(self.batch_size, self.dec_size), volatile=True)

        for t in xrange(self.T):
            if use_cuda:
                c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)).cuda() if t == 0 else self.Cs[t - 1]
            else:
                c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.Cs[t - 1]

            z = self.normalSample()
            h_dec, dec_state = self.Decode(z, (h_dec_prev, dec_state))
            self.Cs[t] = c_prev + self.write_linear(h_dec)
            h_dec_prev = h_dec
        imgs = []
        for img in self.Cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs

if use_cuda:

    model = DrawModule(batch_size=batch_size, enc_size=enc_size, dec_size=dec_size, z_size=z_size, \
                       use_att=use_att, T=T,img_size=img_size,N=N).cuda()
    # model = DrawModule(batch_size=batch_size, enc_size=enc_size, dec_size=dec_size, z_size=z_size, \
    #                    use_att=use_att, T=T,img_size=img_size,N=N)
    # model=torch.nn.DataParallel(model).cuda()
else:
    model = DrawModule(batch_size=batch_size, enc_size=enc_size, dec_size=dec_size, z_size=z_size, \
                       use_att=use_att, T=T,img_size=img_size,N=N)




optimizer=optim.Adam(model.parameters(),lr=lrt,betas=(0.5,0.999))
BCEcriterion=nn.BCELoss(size_average=True)
def loss_func(x,mus,sigmas,logsigmas,x_recons):
    kl_terms=[0]*T
    KL=0
    for i in range(T):
        mu_2=mus[i]*mus[i]
        sigma_2=sigmas[i]*sigmas[i]
        logsigma=logsigmas[i]
        # bug in original pytorch code?
        kl_terms[i]=0.5*torch.sum(mu_2+sigma_2-2*logsigma,1)-0.5
        KL+=kl_terms[i]

    # print 'shape of sum KL',KL.size()
    KL_ave=torch.mean(KL)
    BCE=BCEcriterion(x_recons,x)*A*B
    # print 'BCE:',BCE.cpu().data.numpy()
    return KL_ave+BCE,KL_ave,BCE



#train
# global count

def train(epoch):
    train_loss=0
    avg_loss=0
    for batch_idx,(data,_) in enumerate(train_loader):

        model.zero_grad()
        if use_cuda:
            inputs=Variable(data).cuda()
        else:
            inputs=Variable(data)
        mus, sigmas, logsigmas, x_recons=model.forward(inputs)
        loss=loss_func(inputs,mus,sigmas,logsigmas,x_recons)
        loss.backward()
        optimizer.step()
        train_loss+=loss.data[0]
        avg_loss+=loss.cpu().data.numpy()
        count=count+1

        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.data[0] / len(data)))
        if count % 100 == 0:
            print 'Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 100)
            if count % 3000 == 0:
                torch.save(model.state_dict(), 'save/weights_%d.tar' % (count))
                generate_image(count)
            avg_loss = 0

    print('====> Epoch: {} Average Train loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader)))
    #Not a directory
    torch.save(model.state_dict(),'../models/DRAW/'+'DRAW'+str(epoch))

    return

def generate_image(count):
    x = model.generate(batch_size)
    save_image(x,T,B,A,count)

def save_example_image():
    train_iter = iter(train_loader)
    data, _ = train_iter.next()
    img = data.cpu().numpy().reshape(batch_size, 28, 28)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

def test(epoch):
    test_loss=0
    num_batch=0
    for data,_ in test_loader:
        if use_cuda:
            inputs = Variable(data).cuda()
        else:
            inputs = Variable(data)
        mus, sigmas, logsigmas, x_recons = model.forward(inputs)
        loss, KLloss, BCEloss = loss_func(inputs, mus, sigmas, logsigmas, x_recons)
        test_loss+=loss.cpu().data.numpy()
        num_batch+=1
    print('Epoch-{};Test loss:{}'.format(epoch,test_loss/num_batch))

def main():
    count = 0
    avg_loss = 0
    avg_BCEloss=0
    for epoch in range(epoch_num):
        train_loss=0
        start=time.time()
        for batch_idx,(data,_) in enumerate(train_loader):

            model.zero_grad()
            if use_cuda:
                inputs=Variable(data).cuda()
            else:
                inputs=Variable(data)
            mus, sigmas, logsigmas, x_recons=model.forward(inputs)
            # print inputs.size(),mus.size(),sigmas.size(),logsigmas.size(),x_recons.size()
            loss,KLloss,BCEloss=loss_func(inputs,mus,sigmas,logsigmas,x_recons)
            loss.backward()
            optimizer.step()
            train_loss+=loss.data[0]
            avg_loss+=loss.cpu().data.numpy()
            avg_BCEloss+=BCEloss.cpu().data.numpy()
            count=count+1

            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader),
            #         loss.data[0] / len(data)))
            if count % 100 == 0:
                print 'Epoch-{}; Count-{}; loss: {};BCE loss:{}'.format(epoch, count, avg_loss / 100,avg_BCEloss/100)
                if count % 3000 == 0:
                    torch.save(model.state_dict(), 'save/weights_%d.tar' % (count))
                    generate_image(count)
                avg_loss = 0
                avg_BCEloss=0
        end=time.time()
        print('====> Epoch: {} Average Train loss: {:.4f};elapsed time:{:.2f}'.format(
              epoch, train_loss / len(train_loader),end-start))

        # test(epoch)

        # torch.save(model.state_dict(),'../models/DRAW/'+'DRAW'+str(epoch))
    return

if __name__=='__main__':
    # h_dec_test = Variable(torch.randn(batch_size, dec_size)).cuda()
    # gx, gy, delta, gamma, var = model.attention_param(h_dec_test)
    # # print gx, gy, delta, gamma, var
    # Fx, Fy = model.filterbank(gx, gy, delta, var)
    # print torch.sum(Fx)
    # print torch.sum(Fy)
    main()







# x=Variable(images).cuda() if use_cuda else Variable(images)
# mus,sigmas,logsigmas,x_recons=model.forward(x)
# loss_test=loss_func(x,mus,sigmas,logsigmas,x_recons)
# x_recons_tensors=x_recons.data
# grid_recons=torchvision.utils.make_grid(x_recons_tensors)
# imshow(grid_recons)
