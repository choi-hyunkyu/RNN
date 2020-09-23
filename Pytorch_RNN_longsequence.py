import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

'''
랜덤시드 고정
'''
torch.manual_seed(777)

'''
시퀀스 생성
'''
sentence = (
    "if you want to build a ship, don't drum up people together to "
    "collect wood and don't assign them tasks and work, but rather "
    "teach them to long for the endless immensity of the sea."
    )

'''
딕셔너리 생성 및 정수 인코딩
'''
char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}

'''
하이퍼파라미터
'''
dic_size = len(char_dic)
hidden_size = len(char_dic)
sequence_length = 10 #임의의 숫자
learning_rate = 0.1

'''
데이터 세팅
'''
x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + sequence_length + 1]
    print("{} {} {}".format(i, x_str, y_str))
    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

x_one_hot = [np.eye(dic_size)[x] for x in x_data]

x_train = torch.FloatTensor(x_one_hot)
y_train = torch.LongTensor(y_data)

'''
모델 설계
'''
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers = layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias = True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

model = Net(dic_size, hidden_size, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

'''
훈련
'''
nb_epochs = 50
for epoch in range(nb_epochs):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis.view(-1, dic_size), y_train.view(-1))
    loss.backward()
    optimizer.step()

    results = hypothesis.argmax(dim = 2)
    predict_str = ""
    for j, result in enumerate(results):
        if j == 0:
            predict_str += ''.join([char_set[t] for t in result])
        else:
            predict_str += char_set[result[-1]]
    print(predict_str)