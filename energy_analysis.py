import pandas as pd
import numpy as np
import os 

class clr:
    S = '\033[1m' + '\033[95m'
    E = '\033[0m'
    
def create_features(df):
    
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    return df

def compute_analysis(dataset_path):
    
    df = pd.read_csv(dataset_path, n_rows=10)
    print(df)
    
    skip_rows = int(input("Enter the number of rows to skip: "))
    df = pd.read_csv(dataset_path, skiprows=skip_rows, 
                     names = ['date', 'energy_consumption'], header=0,
                     parse_dates=['date'], index_col='date')
    
    unit = input("Do you want the results in kWh or kW? ").strip().lower()
    if unit == 'kw':
        df['energy_consumption'] = df['energy_consumption'] / df.index.to_series().diff().dt.total_seconds() * 3600
    
    df = create_features(df)
    
    results = {
        "Name of dataset": [dataset_path.split('/')[-1]],
        "Time period of dataset": [f"{df.index.min().strftime('%d.%m.%Y')}-{df.index.max().strftime('%d.%m.%Y')}"],
        "Count amount of kWh values": [len(df)],
        "Total energy consumption": [df['energy_consumption'].sum()],
        "Maximum value in dataset": [f"{df['energy_conumption'].max()} on {df['energy_conumption'].idxmax()}"],
        "Minimum value in dataset": [f"{df['energy_conumption'].min()} on {df['energy_conumption'].idxmin()}"],
    }
    
    daily = df.resample('D').sum()
    
    weekdays = df[df['dayofweek'].between(0, 4)]
    weekends = df[df['dayofweek'].between(5, 6)]
    sun_3am = df[(df['dayofweek'] == 6) & (df['hour'] == 3)]
    weekdays_8_17 = weekdays[weekdays['hour'].between(8, 16)]
    weekends_8_17 = weekends[weekends['hour'].between(8, 16)]
    weekdays_17_8 = weekdays[~weekdays['hour'].between(8, 16)]
    weekends_17_8 = weekends[~weekends['hour'].between(8, 16)]
    
    datasets = {
        "Average and median of daily sum in dataset": daily,
        "Average and median of daily sum on weekdays": weekdays.resample('D').sum(),
        "Average and median of daily sum on saturdays and sundays": weekends.resample('D').sum(),
        "Average and median of on sundays at 03:00 in the morning": sun_3am,
        "Average daily sum between 08:00-17:00 on weekdays": weekdays_8_17.resample('D').sum(),
        "Average daily sum between 08:00-17:00 on Saturdays and sundays": weekends_8_17.resample('D').sum(),
        "Average daily sum between 17:00-08:00 on weekdays": weekdays_17_8.resample('D').sum(),
        "Average daily sum between 17:00-08:00 on saturdays and sundays": weekends_17_8.resample('D').sum(),
    }
    
    for k, v in datasets.items():
        results[k]= [
            v['energy_consumption'].mean(), 
            v['energy_consumption'].median(),
            v['energy_consumption'].std()
        ]
        
    results_df = pd.DataFrame(results).T
    results_df.columns = ['Value']
    
    print(clr.S+'Results'+clr.E)
    print(results_df)
    
    save_path = os.path.join(os.path.dirname(dataset_path), os.path.basename(dataset_path).replace('.csv', '_results.csv'))
    results_df.to_csv(save_path)
    
    print(clr.S+f"Results saved to {save_path} !"+clr.E)

if __name__ == '__main__':
    
    dataset_path = input("Enter the path to the dataset: ")
    compute_analysis(dataset_path)