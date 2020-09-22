# charseq : 변형과 가공이 가능한 문자열

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

'''
랜덤시드 고정
'''
torch.manual_seed(777)

'''
문자열 정의
'''
sample = "if you want you"

'''
단어사전 정의 및 정수인코딩
'''
char_set = list(set(sample))
char_dic = {c: i for i, c in enumerate(char_set)} # c: word, i: index
print(char_dic)

'''
하이퍼파라미터
'''
dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

'''
데이터 세팅
'''
sample_idx = [char_dic[c] for c in sample]
x_data = [sample_idx[:-1]]
x_one_hot = [np.eye(dic_size)[x] for x in x_data]
y_data = [sample_idx[1:]]

'''
데이터 텐서 변환
'''
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

'''
모델, 비용함수, 옵티마이저
'''
rnn = torch.nn.RNN(input_size = dic_size, hidden_size = hidden_size, num_layers = 2, batch_first = True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = learning_rate)

'''
모델 학습
'''
nb_epochs = 50
for i in range(50):
    optimizer.zero_grad()
    outputs, _status = rnn(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)