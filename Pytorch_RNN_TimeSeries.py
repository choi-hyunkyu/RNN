import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(777)
sequence_length = 7 # 7일간의 데이터
input_size = 5 # 1개의 sequence의 내부에 5개의 데이터
hidden_size = 10 # hidden state size가 작으면 data의 압축이 생길 수 있기 때문에 hidden_size를 충분히 크게 해주어야 함
output_size = 1 # 종가 예측
num_layers = 2
learning_rate = 1e-2
nb_epochs = 500

xy = np.loadtxt('data-02-stock_daily.csv', delimiter = ",")
xy = xy[::-1] # 역순 저장

train_size = int(len(xy) * 0.7)
train_set = xy[0: train_size]
test_set = xy[train_size - sequence_length: ]

# scaling function 정의
def minmax_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)

    return numerator / (denominator + 1e-7)

# dataset function 정의
def build_dataset(time_series, sequence_length):
    x_data = []
    y_data = []

    for i in range(0, len(time_series) - sequence_length):
        _x = time_series[i: i + sequence_length, :]
        _y = time_series[i + sequence_length, [-1]]
        print(_x, "->", _y)
        x_data.append(_x)
        y_data.append(_y)

    return np.array(x_data), np.array(y_data)

train_set = minmax_scaler(train_set)
test_set = minmax_scaler(test_set)

x_train, y_train = build_dataset(train_set, sequence_length)
x_test, y_test = build_dataset(test_set, sequence_length)

# convert to tensor
x_train_ts = torch.FloatTensor(x_train)
y_train_ts = torch.FloatTensor(y_train)

x_test_ts = torch.FloatTensor(x_test)
y_test_ts = torch.FloatTensor(y_test)

# 모델설계
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Net, self).__init__()
        self.rnn = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True
        )
        self.fc = nn.Linear(
            hidden_size,
            output_size,
            bias = True
        )

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        return x

model = Net(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(nb_epochs):
    x_train_ts = x_train_ts.to(device)
    y_train_ts = y_train_ts.to(device)
    optimizer.zero_grad()
    hypothesis = model(x_train_ts)
    loss = criterion(hypothesis, y_train_ts)
    loss.backward()
    optimizer.step()

    print("Epoch: {} | Loss: {}".format(epoch, loss.item()))

x_test_ts = x_test_ts.to(device)
plt.plot(y_test)
plt.plot(model(x_test_ts).cpu().data.numpy())
plt.legend(['original', 'prediction'])
plt.show()