import pandas as pd
import os
import sys
from dateutil import parser;


dataFiles = ['TableData (6)', 'TableData (7)', 'TableData (8)']

for i in dataFiles:
    df = pd.read_csv(f"./Dataset/ton_efficiency/efficiency/{i}.csv");

    df['Time'] = pd.to_datetime(df['Time'])

    df.set_index('Time', inplace=True);

    hourly_avg = df.resample('H').mean()

    hourly_avg.to_csv(f'./processed{i}.csv', sep=',', encoding='utf-8')