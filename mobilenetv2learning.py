import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual,self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride==1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim,kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim,hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),

            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv= nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
    


class Model(nn.Module):
    def __init__(self,num_classes=10):
        super(Model,self).__init__()

        self.features = [nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )]

        self.features.append(InvertedResidual(32,16,stride=1,expand_ratio=1))
        self.features.append(InvertedResidual(16,24,stride=2,expand_ratio=6))
        self.features.append(InvertedResidual(24,24,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(24,32,stride=2,expand_ratio=6)) 
        self.features.append(InvertedResidual(32,32,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(32,32,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(32,64,stride=2,expand_ratio=6))
        self.features.append(InvertedResidual(64,64,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(64,64,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(64,64,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(64,96,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(96,96,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(96,96,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(96,160,stride=2,expand_ratio=6))
        self.features.append(InvertedResidual(160,160,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(160,160,stride=1,expand_ratio=6))
        self.features.append(InvertedResidual(160,320,stride=1,expand_ratio=6))

        self.features.append(nn.Sequential(
            nn.Conv2d(320,1280,kernel_size=1,bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        ))

        self.features=nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280,num_classes)
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
