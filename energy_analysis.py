import pandas as pd
import numpy as np
import os 

class clr:
    S = '\033[1m' + '\033[95m'
    E = '\033[0m'
    
def create_features(df):
    """
    Create time series features based on time series index.
    
    Args:
    - df (pd.DataFrame): dataframe with a datetime index.
    
    Returns:
    - pd.DataFrame: Dataframe with additional time-related features.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    return df

def display_results_table(results_df):
    """
    Display a formatted results table.
    
    Args:
    - results_df (pd.DataFrame): Input dataframe containing results.
    """
    
    max_widths = results_df.applymap(lambda x: len(str(x))).max(axis=0)
    column_widths = max_widths.add(5)  # 5 for some padding
    total_width = column_widths.sum() + len(column_widths) + 1  # for borders

    def print_border():
        print('-' * total_width)

    def print_row(items, widths):
        line = "|"
        for item, width in zip(items, widths):
            line += f" {str(item).ljust(width)} |"
        print(line)

    print_border()

    # Print headers
    headers = results_df.columns
    print_row(headers, column_widths)

    print_border()

    # Print data rows
    for _, row in results_df.iterrows():
        values = row.values
        print_row(values, column_widths)
        print_border()
        
def load_data(dataset_path):
    """
    Load energy data from a given path.
    
    Args:
    - dataset_path (str): Path to the dataset.
    
    Returns:
    - pd.DataFrame or None: loaded dataframe or None if an error occurs.
    """
    
    if os.name == 'nt':
        command = f"type {dataset_path} | find /v /c \"\""
    elif os.name == 'posix':
        command = f"head -n 10 {dataset_path}"
    else:
        print(clr.S + "Sorry, this platform is not supported." + clr.E)
        return None

    if os.system(command) != 0:
        print(clr.S + """Error displaying the first 10 rows of the file. 
              Please check the file path and permissions.""" + clr.E)
        return None

    skip_rows = int(input("Enter the number of rows to skip: "))
    try:
        df = pd.read_csv(dataset_path, skiprows=skip_rows,
                         sep=';', decimal=',', dayfirst=True,
                         names=['date', 'energy_consumption'], header=0,
                         parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format='%d.%m.%Y %H:%M', errors='coerce'),
                         index_col='date')

        unit = input("Do you want the results in kWh or kW? ").strip().lower()
        if unit == 'kw':
            df['energy_consumption'] = round(df['energy_consumption'] / df.index.to_series().diff().dt.total_seconds() * 3600, 2)
            
        return df

    except Exception as e:
        print(clr.S + f"Error processing the dataset: {e}" + clr.E)
        return None

def compute_dataset_info(df):
    """
    Compute dataset statistics for different time intervals.

    Args:
        - df (pd.DataFrame): Input dataframe with energy consumption data.

    Returns:
        - pd.DataFrame: Dataframe with computed statistics.
    """
    
    WEEKDAYS = df['dayofweek'].between(0, 4)
    WEEKENDS = df['dayofweek'].between(5, 6)
    WORK_HOURS = df['hour'].between(8, 16)
    OFF_HOURS = ~df['hour'].between(8, 16)

    datasets = {
        "Daily sum": round(df.resample('D').sum(), 2),
        "Weekday sum": round(df[WEEKDAYS].resample('D').sum(), 2),
        "Weekend sum": round(df[WEEKENDS].resample('D').sum(), 2),
        "Sunday at 03:00": round(df[(df['dayofweek'] == 6) & (df['hour'] == 3)], 2),
        "Weekdays 08:00-17:00": round(df[WEEKDAYS & WORK_HOURS].resample('D').sum(),2),
        "Weekends 08:00-17:00": round(df[WEEKENDS & WORK_HOURS].resample('D').sum(), 2),
        "Weekdays 17:00-08:00": round(df[WEEKDAYS & OFF_HOURS].resample('D').sum(), 2),
        "Weekends 17:00-08:00": round(df[WEEKENDS & OFF_HOURS].resample('D').sum(), 2),
    }
    
    dataset_info = {
        'Metric': list(datasets.keys()),
        "Mean": [round(v['energy_consumption'].mean(),2) for v in datasets.values()],
        "Median": [round(v['energy_consumption'].median(),2) for v in datasets.values()],
        "Standard Deviation": [round(v['energy_consumption'].std(),2) for v in datasets.values()]
    }
    
    return pd.DataFrame.from_dict(dataset_info)

def compute_analysis(dataset_path):
    """
    Analyze energy consumption data from a given path.
    
    Args:
        - dataset_path (str): Path to the dataset.
    """
    
    df = load_data(dataset_path)
    
    print(df.head())

    df = create_features(df)

    basic_info = {
     "Name of dataset": dataset_path.split('/')[-1],
     "Time period of dataset": f"{df.index.min()} to {df.index.max()}",
     "Count of kWh values": len(df),
     "Total energy consumption (kWh)": round(df['energy_consumption'].sum(),2),
     "Maximum value in dataset": df['energy_consumption'].max(),
     "Maximum value date": df['energy_consumption'].idxmax(),
     "Minimum value in dataset": df['energy_consumption'].min(),
     "Minimum value date": df['energy_consumption'].idxmin(),
    }

    basic_info_df = pd.DataFrame(list(basic_info.items()), columns=['Metric', 'Value'])

    dataset_info_df = compute_dataset_info(df)

    print(clr.S + 'Basic Information' + clr.E)
    basic_info_df = basic_info_df.reset_index(drop=True)
    display_results_table(basic_info_df)

    print(clr.S + 'Dataset Statistics' + clr.E)
    display_results_table(dataset_info_df)

    basic_info_path = os.path.join(os.path.dirname(dataset_path), os.path.basename(dataset_path).replace('.csv', '_basic_info.csv'))
    dataset_info_path = os.path.join(os.path.dirname(dataset_path), os.path.basename(dataset_path).replace('.csv', '_dataset_info.csv'))

    basic_info_df.to_csv(basic_info_path, index=False)
    dataset_info_df.to_csv(dataset_info_path, index=False)
    print(clr.S + f"Basic Info saved to {basic_info_path} !" + clr.E)
    print(clr.S + f"Dataset Stats saved to {dataset_info_path} !" + clr.E)

if __name__ == '__main__':
    
    dataset_path = input("Enter the path to the dataset: ")
    compute_analysis(dataset_path)