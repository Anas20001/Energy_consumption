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
            if isinstance(item, list):
                
                if not item:
                    item = ''
                    
            elif pd.isna(item):
                item = ''
            
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
            df['energy_consumption'] = round(df['energy_consumption'] / 0.25, 2)
            
        return df, unit

    except Exception as e:
        print(clr.S + f"Error processing the dataset: {e}" + clr.E)
        return None
    
def get_basic_info(df, dataset_path, unit):
    """
    Extracts basic information from the dataset.
    
    Args:
        df (pd.DataFrame): The data frame containing the energy consumption data.
        dataset_path (str): The path to the dataset.
        
    Returns:
        dict: A dictionary containing the extracted basic info.
    """
    total_rows = len(df)
    count_non_nan = df.count()[0]
    
    info = {
        "Name of dataset": dataset_path.split('/')[-1],
        "Time period of dataset": f"{df.index.min()} to {df.index.max()}",
        f"Count of {unit} values": f'{count_non_nan}, 100%',   # this is always 100%
        "Missing data points (NaN)": f'{df.isna().sum()[0]}, {round((df.isna().sum()[0]/total_rows) * 100,2)}%',
        "Missing data points timestamps": df[df['energy_consumption'].isna()].index.tolist(),
        "Count zero values": f'{len(df[df["energy_consumption"] == 0])}, {round((len(df[df["energy_consumption"] == 0])/total_rows) * 100,2)}%',
        "Zero values timestamps": df[df['energy_consumption'] == 0].index.tolist(),
        "Count negative values": f'{len(df[df["energy_consumption"] < 0])}, {round((len(df[df["energy_consumption"] < 0])/total_rows) * 100,2)}%',
        "Negative values": df[df['energy_consumption'] < 0]['energy_consumption'].tolist(),
        "Negative values timestamps": df[df['energy_consumption'] < 0].index.tolist(),
        "Total energy consumption (kWh)": round(df['energy_consumption'].sum(), 2),
        "Maximum value in dataset": df['energy_consumption'].max(),
        "Maximum value date": df['energy_consumption'].idxmax(),
        "Minimum value in dataset": df['energy_consumption'].min(),
        "Minimum value date": df['energy_consumption'].idxmin()
    }
    
    return info

def clean_zero_and_empty_values(info):
    """
    Cleans the values that are zeroes or empty lists.
    
    Args:
        info (dict): The dictionary containing the basic info.
        
    Returns:
        dict: The cleaned dictionary.
    """
    for key, value in info.items():
        if value in ['0, 0.0%', '[]']:
            info[key] = ''
    
    return info

def generate_basic_info_table(df, dataset_path, unit):
    """
    Generates a dictionary containing basic info from the dataset.
    
    Args:
        df (pd.DataFrame): The data frame containing the energy consumption data.
        dataset_path (str): The path to the dataset.
        
    Returns:
        dict: A dictionary containing the basic info.
    """
    info = get_basic_info(df, dataset_path, unit)
    
    return clean_zero_and_empty_values(info)

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

def main(dataset_path):
    
    """
    Analyze energy consumption data from a given path.
    
    Args:
        - dataset_path (str): Path to the dataset.
    """
    
    df, unit = load_data(dataset_path)
    
    rows_to_remove = []

    if unit == 'kw':
        rows_to_remove = [
            'Total energy consumption (kWh)',
            'Daily sum',
            "Weekday sum",
            "Weekend sum",
        ]
    
    df = create_features(df)

    basic_info = generate_basic_info_table(df, dataset_path, unit)

    basic_info_df = pd.DataFrame(list(basic_info.items()), columns=['Metric', 'Value'])
    basic_info_df["Section"] = "Basic Information"

    dataset_info_df = compute_dataset_info(df)
    dataset_info_df["Section"] = "Dataset Statistics"
    
    basic_info_df = basic_info_df[~basic_info_df['Metric'].isin(rows_to_remove)]
    dataset_info_df = dataset_info_df[~dataset_info_df['Metric'].isin(rows_to_remove)]
    
    combined_df = pd.concat([basic_info_df, dataset_info_df], ignore_index=True)
    
    combined_path = os.path.join(os.path.dirname(dataset_path), os.path.basename(dataset_path).replace('.csv', '_combined_info.csv'))
    combined_df.to_csv(combined_path, index=False)

    print(clr.S + 'Combined Information' + clr.E)
    display_results_table(combined_df)

    print(clr.S + f"Combined Info saved to {combined_path} !" + clr.E)
                                
                                
if __name__ == '__main__':
    dataset_path = input("Enter the path to the dataset: ")
    main(dataset_path)