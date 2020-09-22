import torch
import numpy as np

'''
랜덤시드 고정
'''
torch.manual_seed(777)

'''
input, hidden 크기 정의
'''
input_size = 4
hidden_size = 2

h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

input_data_np = np.array([
    [h, e, l, l, o], 
    [e, o, l, l, l], 
    [l, l, e, e, l]], dtype = np.float32
)

rnn = torch.nn.RNN(input_size, hidden_size)

