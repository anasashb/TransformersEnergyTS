import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

def convert_and_save_dataframe(file_path):
    # Ensure that the date column is in datetime format
    input_df = pd.read_csv(file_path)
    input_df['date'] = pd.to_datetime(input_df['date'], format='%Y-%m-%d %H:%M:%S')    

    # Extract day, hour, month, and year from the date
    output_df = pd.DataFrame()
    output_df['value'] = input_df['TARGET']

    standard_scaler = StandardScaler()
    output_df['value'] = standard_scaler.fit_transform(output_df[['value']])
    output_df['day'] = input_df['date'].dt.day
    output_df['hour'] = input_df['date'].dt.hour
    output_df['month'] = input_df['date'].dt.month
    output_df['year'] = input_df['date'].dt.year

    # Apply MinMaxScaler to match covariate scaling
    scaler = MinMaxScaler()
    output_df[['day', 'hour', 'month', 'year']] = scaler.fit_transform(output_df[['day', 'hour', 'month', 'year']])


    output_df = output_df.astype(np.float64).to_numpy()
    output_df = output_df.reshape(1, -1, 5)  # Reshaping to (1, 17420, 5)

    np.save('data/prep_synth_additive.npy', output_df)

if __name__ == "__main__":
    datadir = os.path.join('..', 'SYNTHDataset/SYNTH_additive.csv')
    convert_and_save_dataframe(datadir)