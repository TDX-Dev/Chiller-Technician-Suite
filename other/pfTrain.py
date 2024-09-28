import torch
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pumpFrequencyModel import PlantData, PumpFrequencyModel
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

features = ['month', 'season', 'weekday', 'time_of_day', 'RT', 'kW_Tot', 'kW_CHH', 'DeltaCHW', 'DeltaCDW']
targets = ['CH Load']

df2 = pandas.read_csv("../Dataset/ton_efficiency/efficiency/TableData (8).csv")
df1 = pandas.read_csv("../Dataset/ton_efficiency/efficiency/TableData (7).csv")
df = pandas.read_csv("../Dataset/ton_efficiency/efficiency/TableData (6).csv")

df = pandas.concat([df, df1, df2])

df['Time'] = pandas.to_datetime(df['Time'])

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

X = df[features].to_numpy();
Y = df[targets].to_numpy();



X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state = 42)

print(X_Train.shape, Y_Train.shape)
print(X_Test.shape, Y_Test.shape)
 

# scaler_x = StandardScaler()
# scaler_y = StandardScaler()

# X_Train = scaler_x.fit_transform(X_Train)
# X_Test = scaler_x.transform(X_Test)

# Y_Train = scaler_y.fit_transform(Y_Train)
# Y_Test = scaler_y.transform(Y_Test)

Train_Dataset = PlantData(X_Train, Y_Train, device='cpu');
Test_Dataset = PlantData(X_Test, Y_Test, device='cpu');

Train_Dataloader = DataLoader(Train_Dataset, batch_size=256, shuffle=True);
Test_DataLoader = DataLoader(Test_Dataset, batch_size=256);

model = PumpFrequencyModel(len(features), len(targets));
model.to(device)
print(device)

criterion = nn.MSELoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0001)

def train(training_epochs, model, device):
    for epoch in range(training_epochs):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch in Train_Dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            outputs = model(x_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        test_loss, rmse_loss = validation(model)

        if ((epoch + 1) % 10 == 0):
            print(f'Epoch [{epoch+1}/{training_epochs}], Loss: {epoch_loss/len(Train_Dataloader):.4f}, Test Loss: {test_loss:.4f}, RMSE Loss: {rmse_loss}')


def validation(model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in Test_DataLoader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            test_loss += loss.item()
    return test_loss, mean_squared_error(y_batch.cpu(), predictions.cpu())

    

train(1000, model, device)


# Example new input data (replace with actual values for the features)
new_data = [df[features].iloc[0]]

# # Normalize new data using the same scaler used for training
# new_data_normalized = scaler_x.transform(new_data)

# Convert the data into a PyTorch tensor
new_data_tensor = torch.tensor(new_data, dtype=torch.float32).to(device)

# Switch model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(new_data_tensor)

# Convert predictions back to original scale using inverse transform of the target scaler
predictions_original_scale = predictions.cpu().numpy()

# Print the result
print(f'Predicted Chiller Load: {predictions_original_scale}')

torch.save(model.state_dict(), './pump_freq.pt')
