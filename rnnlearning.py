import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Model(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, num_classes):
        super(Model,self).__init__()
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,num_classes)

        

    def forward(self, x):
        x = x.squeeze(1)
        out,_ = self.rnn(x)
        out = out[:,-1, : ]
        out = self.fc(out)
        return out

#학습함수
def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss=0
        for data, target in train_loader:
            optimizer.zero_grad()
            output=model(data)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

#평가함수
def eval(model,test_loader):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for data,target in test_loader:
            output= model(data)
            _,predicted = torch.max(output,1)
            correct+=(predicted==target).sum().item()
            total+=target.size(0)
    accuracy=100*correct/total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 하이퍼파라미터
input_size = 28  # Each row of the image is treated as a sequence step
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 32
learning_rate = 0.01
epochs = 10

# MNIST 데이터셋 로드, 데이터로드부분은 지피티썻음, 사실나머지도 지피티따라치기임
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#모델, 손실함수, 최적화 정의    
model= Model(input_size, hidden_size, num_layers, num_classes)
crit = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=learning_rate)

train(model,train_loader,opt,crit,epochs)
eval(model,test_loader)
