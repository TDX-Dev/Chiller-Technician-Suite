import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PumpFrequencyModel(nn.Module):

    def __init__(self, features, targets):
        pass
        super(PumpFrequencyModel, self).__init__()
        self.fc1 = nn.Linear(features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, targets)

    def forward(self, x):
        x = torch.relu(self.fc1(x));
        x = torch.relu(self.fc2(x));
        x = self.fc3(x);
        return x


class PlantData(Dataset):
    def __init__(self, X, Y, device):
        self.x = torch.from_numpy(X).type(dtype=torch.float32)
        self.y = torch.from_numpy(Y).type(dtype=torch.float32)
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        return self.x[idx], self.y[idx]