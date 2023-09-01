import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.express import imshow
import pandas as pd
import base64
from io import StringIO

class DataLoader:
    
    def __init__(self, uploaded_file, unit):
        self.uploaded_file = uploaded_file
        self.unit = unit
        

    def preview_data(self, preview_lines):
        """
        preview the loaded data 
        """
        preview = self.uploaded_file.getvalue().decode('ISO-8859-1').split('\n')[:preview_lines]
        return pd.read_csv(StringIO('\n'.join(preview)), sep=';', engine='python')
        
    def try_loadding(self, skip_rows, usecols, columns, consumption_column):
        
        df = pd.read_csv(self.uploaded_file, sep=';', engine='python', decimal=',', 
                                         dayfirst=True, skiprows=skip_rows, header=None, 
                                         encoding='ISO-8859-1', usecols=usecols)

        if len(df.columns) != len(columns):
                st.sidebar.warning("Number of columns in the dataset does not match the number of columns specified by the user.")
                return None
        return df
   
    def load_data(self, column_names, skip_rows, usecols):
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
            df =  self.try_loadding(
                skip_rows, usecols, columns, consumption_column
            )
        except Exception as e:
                st.sidebar.warning(f"Error loading the dataset: {e}")

        # Rename columns based on user input
        df.columns = columns
        if 'date' in columns and 'time' in columns:
            df['timestamp'] = df['date'] + ' ' + df['time']
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.set_index('timestamp')
            df = df[[consumption_column]]

        elif 'timestamp' in columns or 'date' in columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.set_index('timestamp')


        else:
            st.sidebar.warning("Invalid columns format specified by the user. use (date,time,consumption) or (timestamp,consumption))")
            return None

        col_type = df.dtypes[consumption_column]
        if col_type == 'object':
            df[consumption_column] = df[consumption_column].str.split(',').str[0].astype(float)


        if self.unit == 'kw':
                df[consumption_column] = df[consumption_column] / 0.25

        return df, consumption_column



class DataManipulation:        
    
    @staticmethod
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
        df['day'] = df.index.day
    
        return df

class Analyzer:
    def __init__(self, df, unit, consumption_column, dataset_path):
        self.df = df 
        self.unit = unit 
        self.consumption_column = consumption_column
        self.dataset_path = dataset_path


    def generate_basic_info_table(self):
            """
            Generates a DataFrame containing basic info from the dataset.
            
            Args:
                df (pd.DataFrame): The data frame containing the energy consumption data.
                unit (str): Unit of measurement for consumption.
                consumption_column (str): The column name for consumption.
                
            Returns:
                pd.DataFrame: A DataFrame containing the basic info.
            """
            info = self.get_basic_info()
            
            if not isinstance(info, dict):
                raise ValueError(f"get_basic_info() returned a {type(info)}. Expected a dictionary.")
            
            cleaned_info =Utils.clean_zero_and_empty_values(info)
        
            return pd.DataFrame(
                list(cleaned_info.items()), columns=['Metric', 'Value']
            ).dropna()
            
            
    def generate_combined_info_table(self):
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
            if self.unit == 'kw':
                rows_to_remove = [
                    'Total energy consumption (kWh)',
                    'Daily sum',
                    'Weekday sum',
                    'Weekend sum',
                ]
        
            basic_info_df = self.generate_basic_info_table()
            basic_info_df['Section'] = 'Basic Information'
        
            dataset_info_df = self.compute_dataset_info()
            dataset_info_df['Section'] = 'Dataset Statistics'
        
            basic_info_df = basic_info_df[~basic_info_df['Metric'].isin(rows_to_remove)]
            dataset_info_df = dataset_info_df[~dataset_info_df['Metric'].isin(rows_to_remove)]
        
            return pd.concat([basic_info_df, dataset_info_df], ignore_index=True).fillna('')
        

    def compute_dataset_info(self):
            """
            Compute dataset statistics for different time intervals.
        
            Args:
                - df (pd.DataFrame): Input dataframe with energy consumption data.
        
            Returns:
                - pd.DataFrame: Dataframe with computed statistics.
            """
            df = DataManipulation.create_features(self.df)
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
                "Mean": [round(v[self.consumption_column].mean(),2) for v in datasets.values()],
                "Median": [round(v[self.consumption_column].median(),2) for v in datasets.values()],
                "Standard Deviation": [round(v[self.consumption_column].std(),2) for v in datasets.values()]
            }
            
            return pd.DataFrame.from_dict(dataset_info)

        
    def get_basic_info(self):
            # sourcery skip: inline-immediately-returned-variable
            """
            Extracts basic information from the dataset.
            
            Args:
                df (pd.DataFrame): The data frame containing the energy consumption data.
                dataset_path (str): The path to the dataset.
                
            Returns:
                dict: A dictionary containing the extracted basic info.
            """
            total_rows = len(self.df)
            count_non_nan = self.df.count()[0]
            
            zero_values_df = self.df[self.df[self.consumption_column] == 0]
            zero_values_time_range = f"{zero_values_df.index.min()} to {zero_values_df.index.max()}"
            zero_count_by_day = zero_values_df.resample('D').size()
            percentage_zeros = round((len(zero_values_df) / total_rows) * 100, 2)
            
            info = {
                "Name of dataset": self.dataset_path.split('/')[-1],
                "Time period of dataset": f"{self.df.index.min()} to {self.df.index.max()}",
                f"Count of {self.unit} values": f'{count_non_nan}, 100%',  
                "Missing data points (NaN)": f'{self.df.isna().sum()[0]}, {round((self.df.isna().sum()[0]/total_rows) * 100,2)}%',
                
                "Count zero values": f'{len(self.df[self.df[self.consumption_column] == 0])}, {round((len(self.df[self.df[self.consumption_column] == 0])/total_rows) * 100,2)}%',
                "Count zero values": f'{len(zero_values_df)}, {percentage_zeros}%',
                "Zero values time range": zero_values_time_range,
                "Zero values by day": zero_count_by_day.to_dict(),
                "Count negative values": f'{len(self.df[self.df[self.consumption_column].astype(float) < 0])}, {round((len(self.df[self.df[self.consumption_column].astype(float) < 0])/total_rows) * 100,2)}%',
                "Negative values": self.df[self.df[self.consumption_column] < 0][self.consumption_column].tolist(),
                "Negative values timestamps": self.df[self.df[self.consumption_column] < 0].index.tolist(),
                "Total energy consumption (kWh)": round(self.df[self.consumption_column].sum(), 2),
                "Maximum value in dataset": self.df[self.consumption_column].max(),
                "Maximum value date": self.df[self.consumption_column].idxmax(),
                "Minimum value in dataset": self.df[self.consumption_column].min(),
                "Minimum value date": self.df[self.consumption_column].idxmin()
            }
            
            return info
 
        
class Visualizer:

    def __init__(self, df, unit, order):
        self.df = df 
        self.unit = unit 
        self.order = order 
        self

    def plot_daily(self, lp_daily):
            datetime_col = Utils.get_datetime_column(lp_daily)
            if not datetime_col:
                st.warning("No datetime column found in the dataset for daily plotting.")
                return
        
            if self.order == "Descending":
                lp_daily = lp_daily.sort_values(by='consumption', ascending=False)
        
            fig = px.bar(lp_daily, x=datetime_col, y='consumption',
                         title='Daily Electricity Consumption',
                         labels={'consumption':f'Energy Consumption ({self.unit})',
                                 datetime_col: 'Date'},
                         color = 'month',
                         category_orders={"month": list(pd.date_range(lp_daily.index.min(), 
                                                                      lp_daily.index.max(), 
                                                                      freq='D').day_name())},
                         template='none',
                         width=2000, height=800)
        
            fig.update_xaxes(tickangle=45)
        
            st.plotly_chart(fig)
        

    def plot_weekly(self, lp_weekly):
        
            if self.order == "Descending":
                lp_weekly = lp_weekly.sort_values(by='consumption', ascending=False)
        
            fig = px.bar(lp_weekly, x='year_week', y='consumption',
                         title='Weekly Electricity Consumption',
                         labels={'consumption':f'Energy Consumption ({self.unit})',
                                 'year_week': 'Week'},
                         color='year-month',
                         template='none',
                         width=2000, height=800)
            
            fig.update_xaxes(tickvals=lp_weekly['year_week'].unique(), 
                             tickangle=45, type='category')
            st.plotly_chart(fig)
        
        
    def plot_monthly(self, lp_monthly):
            datetime_col =Utils.get_datetime_column(lp_monthly)
            if not datetime_col:
                st.warning("No datetime column found in the dataset for monthly plotting.")
                return
        
            if self.order == "Descending":
                lp_monthly = lp_monthly.sort_values(by='month-year', ascending=False)
        
            fig = px.bar(lp_monthly, x='month-year', y='consumption', 
                         title='Monthly Electricity Consumption ({self.unit})', 
                         labels={'consumption':'Energy Consumption',
                                 'month-year': 'Month'},
                         color='month',
                         category_orders={"month": list(pd.date_range(lp_monthly.index.min(),
                                                                      lp_monthly.index.max(), 
                                                                      freq='M').month_name())},
                         template='none',
                         width=2000, height=800)
            
            fig.update_xaxes(tickvals=lp_monthly['month-year'].unique(),
                     tickangle=45)
        
            st.plotly_chart(fig)
        
        
    def plot_avg_daily_profile(self, avg_lp):
        
            fig = px.line(avg_lp, x='time', y='consumption', 
                          color='weekday',
                          labels={'consumption': f'Energy Consumption ({self.unit})', 
                                  'time': 'Hour of the Day'},
                          title='Average Daily Profile per Day of the Week',
                          template='none',
                          width=2000, height=800)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig)
        

    def plot_outliers(self):
            self.df['month'] = self.df.index.month
            self.df['year'] = self.df.index.year
            self.df['outlier'] = False
            self.df['month-year'] = self.df.index.strftime('%Y-%m')
        
            df =Utils.mark_outliers(df, 'consumption', ['month', 'year'])
        
            # Plotting
            fig = px.scatter(self.df.reset_index(), x='datetime', y='consumption',
                             color='outlier', color_discrete_map={True: 'red', False: 'blue'},
                             title='Scatter Plot for Outlier Detection',
                             labels={'consumption':f'Energy Consumption ({self.unit})', 
                                     'datetime': 'Time'},
                             template='none',
                             width=2000, height=800)
            fig.update_xaxes(tickangle=45, tickvals=self.df['month-year'].unique())
            st.plotly_chart(fig)
        
        
    def plot_weekday_vs_weekend(self):
            # Calculate daily sum
            self.df['day_sum'] = self.df['consumption'].resample('D').sum()
            self.df['weekday'] = self.df.index.weekday
        
            self.df = Utils.mark_outliers(self.df, 'day_sum')
           
            df_weekday = self.df[self.df['weekday'] < 5]
            df_weekend = self.df[self.df['weekday'] >= 5]
        
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_weekday.index, y=df_weekday['day_sum'],
                                     mode='markers',
                                     name='Weekdays',
                                     marker_color=df_weekday['outlier'].map({True: 'red', False: 'blue'})))
            fig.add_trace(go.Scatter(x=df_weekend.index, y=df_weekend['day_sum'],
                                     mode='markers',
                                     name='Weekends',
                                     marker_color=df_weekend['outlier'].map({True: 'red', False: 'green'})))
            fig.update_layout(title=f'Weekday vs Weekend Daily Energy Consumption ({self.unit}) with Outliers',
                              xaxis_title='Date',
                              yaxis_title='Energy Consumption',
                              template='none',
                              width=2000, height=800)
            st.plotly_chart(fig)
        
        
    def plot_histogram(self):
           
            self.df =Utils.mark_outliers(self.df, 'consumption')
            
            # Separate data into outliers and non-outliers
            outliers = self.df[self.df['outlier'] == True]['consumption']
            non_outliers = self.df[self.df['outlier'] == False]['consumption']
        
            fig = go.Figure()
            
            
            fig.add_trace(go.Histogram(x=non_outliers, name='Non-Outliers', marker_color='blue', opacity=0.7))
            fig.add_trace(go.Histogram(x=outliers, name='Outliers', marker_color='red', opacity=0.7))
        
            
            fig.update_layout(barmode='overlay', 
                              title=f'Histogram of Energy Consumption Data Points ({self.unit})',
                              xaxis_title=f'Energy Consumption ({self.unit})',
                              yaxis_title='Frequency',
                              template='none',
                              width=2000, height=800)
            
            fig.update_traces(opacity=0.6)
            
            st.plotly_chart(fig)
        

    def plot_heatmap(self, month, year):
            """
            Creates a heatmap using Plotly based on the month and year chosen by the user.
            
            Args:
            - data (pd.DataFrame): DataFrame with energy consumption data.
            - month (int): Month number.
            - year (int): Year.
            
            Returns:
            - Plotly Figure
            """
            data = self.df[(self.df['year'] == year) & (self.df['month'] == month)]
            
            heatmap_data = data.pivot_table(index=data['day'], 
                                            columns=data['hour'], 
                                            values='consumption', 
                                            aggfunc='mean')
            
            fig = imshow(heatmap_data, 
                         labels=dict(y="Hour of Day", x="Day of Month", 
                         color="consumption"), 
                         title=f"Average Energy Consumption ({self.unit}) Heatmap for: {int(month)} - {int(year)}",
                         color_continuous_scale="RdYlGn_r")
            
            fig.update_xaxes(side="bottom")
            fig.update_yaxes(tickvals=[10, 20, 30])  
            
            fig.update_layout(
                template="none",
                width=2000, 
                height=800
            )
            
            st.plotly_chart(fig)


class Utils:
    
    def __init__(self, df, unit, order):
        self.df = df 
        self.unit = unit 
        self.order = order
        self.data_daily, self.data_weekly, self.data_monthly, self.avg_daily_profile = self.prepare_data_for_plots()
        
    @staticmethod
    def clean_zero_and_empty_values(info):
        """
            Cleans the values that are zeroes or empty lists.
            
            Args:
                info (dict): The dictionary containing the basic info.
                
            Returns:
                dict: The cleaned dictionary.
            """
        for key, value in info.items():
            if str(value) in {'0', '0.0%', '[]'}:
                info[key] = ''

        return info


    def prepare_data_for_plots(self):
            
            datetime_col = 'timestamp'
            # Daily
            df_daily = self.df.resample('D').sum().reset_index()
            df_daily['month'] = df_daily[datetime_col].dt.month_name()
        
            # Weekly
            df_weekly = self.df.resample('W-Mon').sum().reset_index()
            df_weekly['year_week'] = df_weekly[datetime_col].dt.strftime('%Y-%U')
            df_weekly['week_number'] = df_weekly[datetime_col].dt.isocalendar().week
            df_weekly['year-month'] = df_weekly[datetime_col].dt.strftime('%Y-%m')
        
            # Monthly
            df_monthly = self.df.resample('M').sum().reset_index()
            df_monthly['month-year'] = df_monthly[datetime_col].dt.strftime('%Y-%m')
            df_monthly['month'] = df_monthly[datetime_col].dt.month_name()
        
            # AverageDailyProfile
            self.df['time'] = self.df.index.time
            self.df['weekday'] = self.df.index.day_name()
            avg_daily_profile = self.df.groupby(['weekday', 'time']).mean().reset_index()
        
            return df_daily, df_weekly, df_monthly, avg_daily_profile
        

    @staticmethod
    def get_datetime_column(self):
        """Identify the datetime column from the dataframe."""
        return next(
            (col for col in self.df.columns if self.df[col].dtype == 'datetime64[ns]'), None
        )
        

    def mark_outliers(self, target_col='consumption', groupby_cols=None):
        # sourcery skip: extract-method
            """
            Mark outliers in the dataframe based on IQR method for the target column.
            
            Args:
            - df (pd.DataFrame): DataFrame to process.
            - target_col (str): Column name to check for outliers.
            - groupby_cols (list): Columns to group by when calculating IQR. If empty, the IQR is calculated for the entire dataset.
            
            Returns:
            - pd.DataFrame: DataFrame with an added 'outlier' column.
            """
            if groupby_cols is None:
                groupby_cols = []
            self.df['outlier'] = False
        
            # If no groupby columns are provided, calculate IQR for the entire dataset
            if not groupby_cols:
                Q1 = self.df[target_col].quantile(0.25)
                Q3 = self.df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.df['outlier'] = (self.df[target_col] < lower_bound) | (self.df[target_col] > upper_bound)
                return self.df
        
            # If groupby columns are provided, loop through each group and calculate IQR
            for name, group in self.df.groupby(groupby_cols):
                Q1 = group[target_col].quantile(0.25)
                Q3 = group[target_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
        
                mask = (self.df[target_col] < lower_bound) | (self.df[target_col] > upper_bound)
                for col, value in zip(groupby_cols, (name if isinstance(name, tuple) else [name])):
                    mask = mask & (self.df[col] == value)
        
                self.df.loc[mask, 'outlier'] = True
        
            return self.df
        
        
    def show_icons():
            
            order = st.sidebar.radio("Order data by:", ["Ascending", "Descending"])
            plot_type = st.sidebar.radio("Choose the analysis type:", ["None", "Daily", 
                                                                       "Weekly", "Monthly", 
                                                                       "Average Daily Load Profile"], key='plot_type')
            plausibility_check = st.sidebar.radio("Choose a plausibility Check:",
                                                    ['None', 'Outliers Detection',
                                                     'Weekday vs Weekend',
                                                     'Histogram of data points'])
            heatmap = st.sidebar.radio("Choose a heatmap:", ['None', 'Heatmap'])
            
            return order, plot_type, plausibility_check, heatmap

        
    def handle_heatmap(self):
        df = self.df.copy()
        df
        available_years = list(self.df['year'].unique())
        selected_year = st.sidebar.selectbox('Select Year for Heatmap:', available_years)        
        available_months = list(self.df[self.df['year'] == selected_year]['month'].unique())
        selected_month = st.sidebar.selectbox('Select Month for Heatmap:', available_months)
        Visualizer.plotly_heatmap(self.df, selected_month, selected_year, self.unit)
    
    
    def handle_plotting_and_analysis(self,Visualizer, plot_type, plausibility_check, heatmap):
        
        
        plot_type_switch = {
            "Daily": lambda: Visualizer.plot_daily(self.data_daily, self.order, self.unit),
            "Weekly": lambda: Visualizer.plot_weekly(self.data_weekly, self.order, self.unit),
            "Monthly": lambda: Visualizer.plot_monthly(self.data_monthly, self.order, self.unit),
            "Average Daily Load Profile": lambda: Visualizer.plot_avg_daily_profile(self.avg_daily_profile, self.order, self.unit)
        }
    
        plausibility_check_switch = {
            'Outliers Detection': lambda: Visualizer.plot_outliers(self.df, self.unit),
            'Weekday vs Weekend': lambda: Visualizer.plot_weekday_vs_weekend(self.df, self.unit),
            'Histogram of data points': lambda: Visualizer.plot_histogram(self.df, self.unit)
        }
    
        if plot_type in plot_type_switch:
            plot_type_switch[plot_type]()
        
        if plausibility_check in plausibility_check_switch:
            plausibility_check_switch[plausibility_check]()
        
        if heatmap != 'None':
            self.handle_heatmap()     