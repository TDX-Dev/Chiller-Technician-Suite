import numpy as np
import pandas as pd
import os

print(os.listdir())
df = pd.read_csv("./ton_efficiency/efficiency/TableData (6).csv")

features = ['kW_Tot', 'RT', 'CH Load']
targets = ['Hz_ CHP', 'Hz_CHS', 'Hz_CT']

correlation_matrix = np.corrcoef(df[features].to_numpy().flatten(), df[targets].to_numpy().flatten())
print(correlation_matrix)