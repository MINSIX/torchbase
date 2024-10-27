import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model,self).__init__()
        def depthwise_separable_conv(in_channels,out_channels,stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True)
            )
        self.features = nn.Sequential(
            depthwise_separable_conv(1,32,stride=2),
            depthwise_separable_conv(32,64,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),

            depthwise_separable_conv(64,128,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            depthwise_separable_conv(128,256,stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*1*1,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

# 학습 함수
def train(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# 평가 함수
def eval(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 하이퍼파라미터
batch_size = 32
learning_rate = 0.001
epochs = 10

# GPU 사용 가능 여부 체크
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# MNIST 데이터셋 로드
transform = transforms.Compose([
    transforms.Resize(28),  # MNIST는 원래 28x28 사이즈를 유지
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize 적용
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 최적화 정의
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 평가
train(model, train_loader, optimizer, criterion, epochs, device)
eval(model, test_loader, device)
