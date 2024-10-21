import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



#모델 정의
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 채널: 1, 출력 채널: 32, 필터 크기: 3x3
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        # 출력 채널: 64
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(2,2)# 맥스풀링 2x2
        # 완전 연결층 (입력: 64x7x7, 출력: 128)
        self.fc1=nn.Linear(64*7*7,128)
        # 출력 클래스: 10 (0-9)
        self.fc2=nn.Linear(128,10)

    def forward(self, x):
        #conv1 -> Relu -> Maxpooling
        x = self.pool(F.relu(self.conv1(x)))
        #conv1 -> Relu -> Maxpooling-> conv2 -> relu-> maxpooling
        x = self.pool(F.relu(self.conv2(x)))
        #faltten, view로도 해줄수잇음
        x = torch.flatten(x, start_dim=1)
        #fclayer
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x

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

#하이퍼파라미터
batch_size = 32
learning_rate=0.01
epochs=10

# MNIST 데이터셋 로드, 데이터로드부분은 지피티썻음, 사실나머지도 지피티따라치기임
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#모델, 손실함수, 최적화 정의    
model= Model()
crit = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=learning_rate)

train(model,train_loader,opt,crit,epochs)
eval(model,test_loader)
