import torch
import torch.optim as optim
import numpy as np

'''
랜덤시드 고정
'''
torch.manual_seed(777)

'''
문자열 생성
'''
char_set = ['h', 'i', 'e', 'l', 'o']

'''
하이퍼파라미터
'''
input_size = len(char_set)
hidden_size = len(char_set)
learning_rate = 0.1

'''
데이터 세팅
'''
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]

'''
텐서 변환
'''
x_train = torch.FloatTensor(x_one_hot)
y_train = torch.LongTensor(y_data)

'''
모델 정의
'''
rnn = torch.nn.RNN(
    input_size = input_size,
    hidden_size = hidden_size,
    batch_first = True
    )

'''
비용함수, 옵티마이저 정의
'''
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = learning_rate)

'''
학습
'''
nb_epochs = 50
for epoch in range(nb_epochs):
    optimizer.zero_grad()
    outputs, _status = rnn(x_train)
    loss = criterion(outputs.view(-1, input_size), y_train.view(-1))
    loss.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis = 2)
    result_str = ''.join([char_set[c] for c in np.squeeze(result)])
    print("Epoch: {}/{} | Loss: {} | Prediction: {} | Prediction_str".format(
        epoch + 1, nb_epochs,
        loss.item(),
        result,
        y_data,
        result_str
    ))