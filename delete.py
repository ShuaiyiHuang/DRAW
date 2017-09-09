import numpy as np
import torch
from torch.autograd import Variable

N=5
A=10
batch_size=3
var=1
mu=Variable(torch.unsqueeze(torch.arange(0,N),0).expand(batch_size,N))
delta=Variable(torch.ones(batch_size,1))
gx=Variable(torch.ones(batch_size,1))
mu_squeezed=(mu-N*0.5-0.5)*delta+gx
mu=torch.unsqueeze(mu_squeezed,2).expand(batch_size,N,A)
Fx_init=Variable(torch.unsqueeze(torch.unsqueeze(torch.arange(0,A),0).expand(N,A),0).expand(batch_size,N,A))
Fx=torch.exp(-(Fx_init-mu)*(Fx_init-mu)/(2*var))
print mu