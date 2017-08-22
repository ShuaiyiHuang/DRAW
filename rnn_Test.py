#RNN

import torch
import torch.nn as nn
from torch.autograd import Variable

batch_size=3
input_size=784
enc_size=5

input_x=Variable(torch.randn(batch_size,input_size))

encoder=torch.nn.LSTMCell(input_size=input_size,hidden_size=enc_size)

h_pre_state=Variable(torch.zeros(batch_size,enc_size))
cell_pre_state=Variable(torch.zeros(batch_size,enc_size))

for i in range(3):
    h_pre_state,cell_pre_state=encoder(input_x,(h_pre_state,cell_pre_state))

print h_pre_state
print cell_pre_state