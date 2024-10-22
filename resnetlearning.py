import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Residual Block 정의
class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1):
        super(ResidualBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.shortCut=nn.Sequential()
        if stride!= 1 or in_channels!=out_channels:
            self.shortCut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortCut(x)
        out=F.relu(out)
        return out
#모델 정의
class Model(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        super(Model,self).__init__()
        self.in_channels=64
        self.conv1=nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.layer1=self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2=self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3=self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4=self._make_layer(block,512,num_blocks[3],stride=2)
        self.fc=nn.Linear(512,num_classes)

    def _make_layer(self,block,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels=out_channels
        return nn.Sequential(*layers)
        

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=F.avg_pool2d(out,4)
        out=torch.flatten(out,1)
        out=self.fc(out)
        return out
    
def Model18():
    return Model(ResidualBlock, [2, 2, 2, 2])
#학습함수

# Modify train function to use device
def train(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # Move data and target to GPU
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Modify eval function to use device
def eval(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # Move data and target to GPU
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = 100 * correct / total
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


# Move model to the selected device (GPU or CPU)
model = Model18().to(device)

crit = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=learning_rate)

train(model,train_loader,opt,crit,epochs,device)
eval(model,test_loader,device)
