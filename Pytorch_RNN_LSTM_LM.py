import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sentence = "오늘은 추석입니다. 모두 가족과 즐거운 시간 보내세요."

x = sentence[:-1] # input
y = sentence[1:] # output

char_set = list(set(sentence))
input_size = len(char_set)
hidden_size = len(char_set)

index2char = {i: c for i, c in enumerate(char_set)}
char2index = {c: i for i, c in enumerate(char_set)}

one_hot = []
for i, tkn in enumerate(x):
    one_hot.append(np.eye(len(char_set), dtype = 'int')[char2index[tkn]])

x_train = torch.Tensor(one_hot)
x_train = x_train.view(1, len(x), -1)

y_label = [char2index[c] for c in y]
y_label = torch.Tensor(y_label)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = 4,
            batch_first = True,
            bidirectional = True # bidirectional을 True로 했기 때문에 마지막 output의 형태는 input_size*2의 형태
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_size * 2, hidden_size)
        )

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.layers(y)
        return y

model = RNN(input_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

nb_epochs = 5000
for epoch in range(nb_epochs):
    x_train = x_train.to(device)
    y_label = y_label.to(device)
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis.view(-1, input_size), y_label.view(-1).long())
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        result = hypothesis.data.cpu().numpy().argmax(axis = 2)
        result_str = ''.join([char_set[c] for c in np.squeeze(result)])
        print(epoch, "loss: ", loss.item(), "\nprediction: ", result, "\ntrue Y: ", y_label, "\nprediction str: ", result_str,"\n")