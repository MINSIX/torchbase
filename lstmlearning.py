import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchaudio
from torchaudio.datasets import GTZAN
import os
import numpy as np

class MelSpec(Dataset):
    def __init__(self, root, transform=None, download=False):
        self.dataset = GTZAN(root=root, download=download)
        self.transform = transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram()
        
        # GTZAN 데이터셋의 장르와 인덱스를 수동으로 매핑
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                       'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.genre_to_index = {genre: idx for idx, genre in enumerate(self.genres)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, label = self.dataset[idx]  # label은 문자열로 제공됨
        mel_spec = self.mel_transform(waveform)
        
        # 정규화 적용
        if self.transform:
            mel_spec = self.transform(mel_spec)

        # 문자열 레이블을 정수 인덱스로 변환
        label_idx = self.genre_to_index[label]
        return mel_spec, torch.tensor(label_idx)

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        # CNN Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # Fully Connected Layer
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)

        # Reshape for LSTM
        x = x.permute(0, 2, 3, 1)  # batch, freq, time, channel
        x = x.contiguous().view(x.size(0), x.size(2), -1)  # (batch, time, features), features=2048

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 마지막 타임 스텝의 출력을 사용
        x = self.fc(x)
        return x

    
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


# 데이터 준비
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.Normalize((0.5,), (0.5,))])
dataset = MelSpec(root='data', transform=transform)

# Train-Test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 100

# 모델 학습
train(model, train_loader, optimizer, criterion, epochs, device)

# 모델 평가
eval(model, test_loader, device)
