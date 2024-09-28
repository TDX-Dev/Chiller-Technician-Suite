import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

# Custom Dataset class for time series data
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_window=24*6, output_window=1):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.input_window]
        y = self.data[idx+self.input_window:idx+self.input_window+self.output_window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Model: LSTM for Time Series Forecasting
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_window):
        super(LSTMTimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * output_window)
        self.output_window = output_window
        self.output_size = output_size

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get last output from LSTM sequence
        out = self.fc(lstm_out)
        out = out.view(-1, self.output_window, self.output_size)  # Reshape to (batch_size, output_window, output_size)
        return out


# Hyperparameters
input_size = 4  # Kw_Tot, Kw_RT, ch_total, Hz_CHS (excluding 'Time')
hidden_size = 64
num_layers = 2
output_size = 1  # Predicting for Kw_Tot, Kw_RT, ch_total, Hz_CHS
input_window = 6 * 24  # 10 minutes per hour, 24 hours (one day sequence)
output_window = 6  # Predict next 1 hour
batch_size = 128
epochs = 150

scaler = MinMaxScaler()

# Load and preprocess data (assumed in pandas DataFrame)
# Example: 'data' should be a DataFrame with columns ['Time', 'Kw_Tot', 'Kw_RT', 'ch_total', 'Hz_CHS']
def preprocess_data(df):
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    df['hour'] = df['Time'].dt.hour

    # Classify hours into morning, afternoon, evening, and night
    df['time_of_day'] = pd.cut(df['hour'], 
                                bins=[-1, 5, 11, 17, 24], 
                                labels=[3, 0, 1, 2])  # Night=3, Morning=0, Afternoon=1, Evening=2

    # Classify weekends and weekdays
    df['weekday'] = df['Time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)  # 1 if weekend, 0 if weekday

    # Classify months into seasons
    df['month'] = df['Time'].dt.month
    df['season'] = df['month'].map({
        12: 0,  # December
        1: 0,   # January
        2: 0,   # February
        3: 1,   # March
        4: 1,   # April
        5: 1,   # May
        6: 2,   # June
        7: 2,   # July
        8: 2,   # August
        9: 3,   # September
        10: 3,  # October
        11: 3   # November
    })
    
    scaled_data = scaler.fit_transform(df[['month', 'season', 'weekday', 'time_of_day', 'RT', 'kW_Tot', 'kW_CHH', 'DeltaCHW', 'DeltaCDW', 'CH Load']])
    scaled_df = pd.DataFrame(scaled_data, columns=['month', 'season', 'weekday', 'time_of_day', 'RT', 'kW_Tot', 'kW_CHH', 'DeltaCHW', 'DeltaCDW', 'CH Load'])

    data = scaled_df[['month', 'season', 'weekday', 'time_of_day', 'RT', 'kW_Tot', 'kW_CHH', 'DeltaCHW', 'DeltaCDW', 'CH Load']].values
    return data

# Example DataFrame
df = pd.read_csv('../Dataset/ton_efficiency/efficiency/TableData (6).csv')  # Replace with actual file
data = preprocess_data(df)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataset and dataloader
train_dataset = TimeSeriesDataset(data, input_window, output_window)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = LSTMTimeSeriesModel(input_size, hidden_size, num_layers, output_size, output_window)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)  # Add batch dimension
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}')


def predict_specific_date(model, input_data, specific_date, steps=6):
    model.eval()
    
    # Ensure specific_date is a datetime object
    specific_date = pd.to_datetime(specific_date)
    
    # Get the index of the specific date and time
    start_idx = np.where(df.index == specific_date)[0][0]
    
    predictions = []
    
    # Prepare the input sequence
    input_seq = input_data[start_idx - input_window:start_idx]  # Last 'input_window' values
    input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        for _ in range(steps):
            input_seq = input_seq.to(device)
            pred = model(input_seq)  # Shape (batch_size, output_window, num_features)
            predictions.append(pred.squeeze(0).cpu().numpy())  # Remove the batch dimension

            # Update input sequence
            new_input = pred[:, -1, :].unsqueeze(1)  # Shape (batch_size, 1, num_features)
            new_input = new_input.to(device)
            
            # Concatenate new input to the sequence and remove the oldest timestep
            input_seq = torch.cat((input_seq[:, 1:, :], new_input), dim=1)

    return np.array(predictions)

# Example usage:
specific_date = '2024-01-01 12:00:00'  # Specify the date and time
steps = 6  # Number of time steps to predict (e.g., the next hour in 10-minute intervals)
predictions = predict_specific_date(model, data, specific_date, steps)

# Process predictions
predictions = predictions.flatten().reshape(-1, 4)
predictions = scaler.inverse_transform(predictions)  # Inverse transform the predictions

# Convert predictions to DataFrame
prediction_df = pd.DataFrame(predictions, columns=['month', 'season', 'weekday', 'time_of_day', 'RT', 'kW_Tot', 'kW_CHH', 'DeltaCHW', 'DeltaCDW', 'CH Load'])
prediction_df['Time'] = pd.date_range(start=specific_date, periods=steps, freq='10T')  # 10-minute intervals

prediction_df.to_csv("predictions.csv", sep=",", index=False)

with open("scaler.json", 'w') as f:
    import json
    json.dump({
        'min_': scaler.data_min_.tolist(),
        'max_': scaler.data_max_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'data_range_': scaler.data_range_.tolist(),
    }, f)