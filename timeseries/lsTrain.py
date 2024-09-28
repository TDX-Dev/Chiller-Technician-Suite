import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lstm_model import TimeSeriesDataset, LSTMModel

# Load the dataset (assuming it's from CSV files)
date_rng = pd.concat([
    pd.read_csv("../Dataset/ton_efficiency/efficiency/TableData (6).csv"),
    pd.read_csv("../Dataset/ton_efficiency/efficiency/TableData (7).csv"),
    pd.read_csv("../Dataset/ton_efficiency/efficiency/TableData (8).csv")
])

# Prepare the data (example columns: adjust as needed)
data = {
    'Time': date_rng['Time'],
    'Kw_Tot': date_rng['kW_Tot'],
    'Kw_RT': date_rng['kW_RT'],
    'ch_total': date_rng['CH Load'],
    'Hz_CHS': date_rng['Hz_CHS']
}
df = pd.DataFrame(data)

# Device setup (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalize the features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Kw_Tot', 'Kw_RT', 'ch_total', 'Hz_CHS']])
scaled_df = pd.DataFrame(scaled_data, columns=['Kw_Tot', 'Kw_RT', 'ch_total', 'Hz_CHS'])

# Parameters
n_steps = 60  # Number of time steps to look back
n_features = scaled_df.shape[1]  # Number of features

# Dataset and DataLoader
dataset = TimeSeriesDataset(scaled_df.values, n_steps)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# LSTM Model setup
model = LSTMModel(n_features).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Constants for future prediction
num_predictions = 365  # Number of future time steps to predict

def predict_future(model, data, sequence_length, num_predictions):
    model.eval()
    predictions = []
    
    # Ensure data is a NumPy array
    data = np.array(data)

    # Use the last `sequence_length` values for the initial sequence
    input_seq = torch.tensor(data[-sequence_length:], dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, sequence_length, num_features)

    for _ in range(num_predictions):
        with torch.no_grad():
            # Predict the next step
            predicted_value = model(input_seq)
            
            # Append the predicted value (assuming prediction has shape: (1, 1, num_features))
            predictions.append(predicted_value.squeeze(0).cpu().numpy())
            
            # Reshape predicted_value to have an additional dimension for sequence length (batch_size, seq_len=1, num_features)
            predicted_value = predicted_value.unsqueeze(1)  # Shape: (batch_size, 1, num_features)
            
            # Use the predicted value as the next input, update the sequence
            input_seq = torch.cat((input_seq[:, 1:, :], predicted_value), dim=1)  # Shift and append the predicted value

    return np.array(predictions)


# Use the last `n_steps` from the dataset to predict the next year
predicted_values = predict_future(model, scaled_df.values, sequence_length=n_steps, num_predictions=num_predictions)

# Inverse scaling the predictions (if needed)
predicted_values_rescaled = scaler.inverse_transform(predicted_values)

print(predicted_values_rescaled)
