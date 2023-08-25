import streamlit as st
import plotly.express as px
import pandas as pd
import base64

st.set_page_config(layout="wide")


## Helper functions 

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

def get_basic_info(df, unit, consumption_column, dataset_path):
    # sourcery skip: inline-immediately-returned-variable
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
        f"Count of {unit} values": f'{count_non_nan}, 100%',  
        "Missing data points (NaN)": f'{df.isna().sum()[0]}, {round((df.isna().sum()[0]/total_rows) * 100,2)}%',
        "Missing data points timestamps": df[df[consumption_column].isna()].index.tolist(),
        "Count zero values": f'{len(df[df[consumption_column] == 0])}, {round((len(df[df[consumption_column] == 0])/total_rows) * 100,2)}%',
        "Zero values timestamps": df[df[consumption_column] == 0].index.tolist(),
        "Count negative values": f'{len(df[df[consumption_column].astype(float) < 0])}, {round((len(df[df[consumption_column].astype(float) < 0])/total_rows) * 100,2)}%',
        "Negative values": df[df[consumption_column] < 0][consumption_column].tolist(),
        "Negative values timestamps": df[df[consumption_column] < 0].index.tolist(),
        "Total energy consumption (kWh)": round(df[consumption_column].sum(), 2),
        "Maximum value in dataset": df[consumption_column].max(),
        "Maximum value date": df[consumption_column].idxmax(),
        "Minimum value in dataset": df[consumption_column].min(),
        "Minimum value date": df[consumption_column].idxmin()
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
        if value in ['0, 0.0%', '[]', []]:
            info[key] = ''

    return info

def generate_basic_info_table(df, unit, consumption_column, dataset_path):
    """
    Generates a DataFrame containing basic info from the dataset.
    
    Args:
        df (pd.DataFrame): The data frame containing the energy consumption data.
        unit (str): Unit of measurement for consumption.
        consumption_column (str): The column name for consumption.
        
    Returns:
        pd.DataFrame: A DataFrame containing the basic info.
    """
    info = get_basic_info(df, unit, consumption_column, dataset_path)
    cleaned_info = clean_zero_and_empty_values(info)

    return pd.DataFrame(
        list(cleaned_info.items()), columns=['Metric', 'Value']
    ).dropna()

def compute_dataset_info(df, consumption_column):
    """
    Compute dataset statistics for different time intervals.

    Args:
        - df (pd.DataFrame): Input dataframe with energy consumption data.

    Returns:
        - pd.DataFrame: Dataframe with computed statistics.
    """
    df = create_features(df)
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
        "Mean": [round(v[consumption_column].mean(),2) for v in datasets.values()],
        "Median": [round(v[consumption_column].median(),2) for v in datasets.values()],
        "Standard Deviation": [round(v[consumption_column].std(),2) for v in datasets.values()]
    }
    
    return pd.DataFrame.from_dict(dataset_info)

def load_data(uploaded_file, column_names, skip_rows, unit):
    """
    Load energy data from a given path.
    
    Args:
    - uploaded_file (UploadedFile): File uploaded through Streamlit.
    - column_names (str): Names of columns separated by commas.
    - skip_rows (int): Number of rows to skip.
    - unit (str): Unit of results ("kWh" or "kW").
    
    Returns:
    - pd.DataFrame or None: loaded dataframe or None if an error occurs.
    """
    
    columns = column_names.split(',')
    
    # Ensure 'consumption' column exists or ask the user for its equivalent
    if 'consumption' not in columns:
        consumption_column = st.sidebar.text_input(f"Specify the column for energy consumption from: {', '.join(columns)}", value='consumption')
        if consumption_column not in columns:
            st.sidebar.warning("Specified column for energy consumption is not among the provided columns.")
            return None
    else:
        consumption_column = 'consumption'
    
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file, sep=';', decimal=',', 
                         dayfirst=True, skiprows=skip_rows, header=None, 
                         encoding='ISO-8859-1')
        
        if len(df.columns) != len(columns):
            st.sidebar.warning("Number of columns in the dataset does not match the number of columns specified by the user.")
            return None

        # Rename columns based on user input
        df.columns = columns
        
        if 'date' in columns and 'time' in columns:
            df['datetime'] = df['date'] + ' ' + df['time']
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.set_index('datetime')
            df = df[[consumption_column]]
        
        elif 'timestamp' in columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.set_index('timestamp')
        
        else:
            st.sidebar.warning("Invalid columns format specified by the user. use (date,time,consumption) or (timestamp,consumption))")
            return None

        if unit == 'kw':
            df[consumption_column] = df[consumption_column] / 0.25
     
        return df
    
    except Exception as e:
        st.sidebar.warning(f"Error loading the dataset: {e}")
        return None

## data preparation functions
def prepare_data_for_plots(df):
    # Daily
    df_daily = df.resample('D').sum().reset_index()
    df_daily['month'] = df_daily['datetime'].dt.month_name()

    # Weekly
    df_weekly = df.resample('W-Mon').sum().reset_index()
    df_weekly['year_week'] = df_weekly['datetime'].dt.strftime('%Y-%U')
    df_weekly['week_number'] = df_weekly['datetime'].dt.isocalendar().week

    # Monthly
    df_monthly = df.resample('M').sum().reset_index()
    df_monthly['month-year'] = df_monthly['datetime'].dt.strftime('%Y-%m')
    df_monthly['month'] = df_monthly['datetime'].dt.month_name()

    # AverageDailyProfile
    df['time'] = df.index.time
    df['weekday'] = df.index.day_name()
    avg_daily_profile = df.groupby(['weekday', 'time']).mean().reset_index()

    return df_daily, df_weekly, df_monthly, avg_daily_profile

def get_datetime_column(df):
    """Identify the datetime column from the dataframe."""
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            return col
    return None  # Return None if no datetime column is found.

## Plots functions 
def plot_daily(lp_daily, order):
    datetime_col = get_datetime_column(lp_daily)
    if not datetime_col:
        st.warning("No datetime column found in the dataset for daily plotting.")
        return

    if order == "Descending":
        lp_daily = lp_daily.sort_values(by='consumption', ascending=False)

    fig = px.bar(lp_daily, x=datetime_col, y='consumption',
                 title='Daily Electricity Consumption',
                 labels={'consumption':'Energy Consumption',
                         datetime_col: 'Date'},
                 color = 'month',
                 category_orders={"month": list(pd.date_range(lp_daily.index.min(), 
                                                              lp_daily.index.max(), 
                                                              freq='D').day_name())},
                 template='plotly_dark',
                 width=2000, height=800)

    fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig)

def plot_weekly(lp_weekly, order):

    if order == "Descending":
        lp_weekly = lp_weekly.sort_values(by='consumption', ascending=False)

    fig = px.bar(lp_weekly, x='year_week', y='consumption',
                 title='Weekly Electricity Consumption',
                 labels={'consumption':'Energy Consumption',
                         'year_week': 'Week'},
                 color='year_week',
                 template='plotly_dark',
                 width=2000, height=800)
    fig.update_xaxes(tickvals=lp_weekly['year_week'].unique(), 
                     tickangle=45, type='category')
    st.plotly_chart(fig)

def plot_monthly(lp_monthly, order):
    datetime_col = get_datetime_column(lp_monthly)
    if not datetime_col:
        st.warning("No datetime column found in the dataset for monthly plotting.")
        return

    if order == "Descending":
        lp_monthly = lp_monthly.sort_values(by='month-year', ascending=False)

    fig = px.bar(lp_monthly, x='month-year', y='consumption', 
                 title='Monthly Electricity Consumption', 
                 labels={'consumption':'Energy Consumption',
                         'month-year': 'Month'},
                 color='month',
                 category_orders={"month": list(pd.date_range(lp_monthly.index.min(),
                                                              lp_monthly.index.max(), 
                                                              freq='M').month_name())},
                 template='plotly_dark',
                 width=2000, height=800)
    
    fig.update_xaxes(tickvals=lp_monthly['month-year'].unique(),
             tickangle=45)
    #fig.update_layout(bargap=0.1, barmode='group')

    st.plotly_chart(fig)

def plot_avg_daily_profile(avg_lp, order):

    fig = px.line(avg_lp, x='time', y='consumption', 
                  color='weekday',
                  labels={'consumption': 'Energy Consumption', 'time': 'Hour of the Day'},
                  title='Average Daily Profile per Day of the Week',
                  template='plotly_dark',
                  width=2000, height=800)
                  
                  
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)

def generate_combined_info_table(df, unit, consumption_column, dataset_path):
    """
    Generate a DataFrame containing basic info and dataset statistics.
    Args:
        df (pd.DataFrame): The data frame containing the energy consumption data.
        unit (str): Unit of measurement for consumption.
        consumption_column (str): The column name for consumption. 
        dataset_path (str): Path to the dataset.
        
    Returns:
        pd.DataFrame: A DataFrame containing the combined info.
    """
    
    rows_to_remove = []
    if unit == 'kw':
        rows_to_remove = [
            'Total energy consumption (kWh)',
            'Daily sum',
            'Weekday sum',
            'Weekend sum',
        ]

    basic_info_df = generate_basic_info_table(df, unit, df.columns[-1], dataset_path)
    basic_info_df['Section'] = 'Basic Information'

    dataset_info_df = compute_dataset_info(df, df.columns[-1])
    dataset_info_df['Section'] = 'Dataset Statistics'

    basic_info_df = basic_info_df[~basic_info_df['Metric'].isin(rows_to_remove)]
    dataset_info_df = dataset_info_df[~dataset_info_df['Metric'].isin(rows_to_remove)]

    return pd.concat([basic_info_df, dataset_info_df], ignore_index=True).fillna('')

def show_icons():
    
    order = st.sidebar.radio("Order data by:", ["Ascending", "Descending"])
    plot_type = st.sidebar.radio("Choose the analysis type:", ["None", "Daily", "Weekly", "Monthly", "Average Daily Load Profile"], key='plot_type')
    return order, plot_type

## Main function
def main():
    st.title("Energy Consumption Analysis")
    st.sidebar.title("Options")

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])    
    
    st.write("Welcome to the Energy Consumption Analysis App!")
    st.write("Please upload a CSV file or use the preloaded LP and select an analysis type to get started.")
    st.image("Landing_page.png")

    if uploaded_file:
        preview_lines = st.sidebar.slider("Select number of lines to preview:", 5, 25, 10)
        column_names = st.sidebar.text_input("Column names (comma-separated)")
        skip_rows = st.sidebar.number_input("Number of rows to skip", step=1)
        unit = st.sidebar.radio("Results unit", ["kWh", "kW"])
        
        st.write(f"## Preview of uploaded data (Top {preview_lines} lines)")
        preview = uploaded_file.getvalue().decode().split('\n')[:preview_lines]
        st.code("\n".join(preview), language='plaintext')
        
        order, plot_type = show_icons()
    
        if column_names and skip_rows is not None and unit:
            df = load_data(uploaded_file, column_names, skip_rows, unit)
            st.write("## Data Sample (Top 5 rows)")
            st.table(df.head())
            
            st.write("## General consumption information")
            st.table(df['consumption'].describe())
            
            combined_df = generate_combined_info_table(df, unit.strip().lower(), df.columns[-1], uploaded_file.name)
            st.write("## Statistical Analysis")
            st.table(combined_df.replace('[]', ''))
            
            csv =combined_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  
            href = f'<a href="data:file/csv;base64,{b64}" download="{uploaded_file.name}_combined_info.csv" style="font-size:20px; color:red; text-decoration:underline;">Download combined info</a>'
            st.markdown(href, unsafe_allow_html=True)

            

            data_daily, data_weekly, data_monthly, avg_daily_profile = prepare_data_for_plots(df)
                        
            if plot_type == "Daily":
                plot_daily(data_daily, order)
            elif plot_type == "Weekly":
                plot_weekly(data_weekly, order)
            elif plot_type == "Monthly":
                plot_monthly(data_monthly, order)
            elif plot_type == "Average Daily Load Profile":
                plot_avg_daily_profile(avg_daily_profile, order)
                

if __name__ == "__main__":
    main()
