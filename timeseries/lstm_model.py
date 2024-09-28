from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

class TimeSeriesDataset(Dataset):
    def __init__(self, data, n_steps):
        self.data = data
        self.n_steps = n_steps

    def __len__(self):
        return len(self.data) - self.n_steps

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.n_steps]
        y = self.data[idx + self.n_steps]
        return torch.FloatTensor(x), torch.FloatTensor(y)
    

class LSTMModel(nn.Module):
    def __init__(self, n_features):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, n_features)  # Output size matches the number of features

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out