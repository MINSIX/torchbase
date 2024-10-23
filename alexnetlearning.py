import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# AlexNet 모델 정의
class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes)
        )

    def forward(self, x):
        x=self.features(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
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
transform = transforms.Compose([
    transforms.Resize(224),  # 이미지 크기 조정
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
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
