import streamlit as st
import plotly.express as px
import pandas as pd

@st.cache_data
def load_daily_data():
    lp_daily = pd.read_csv('data/preprocessed/lp_daily.csv', parse_dates=['date'], index_col=0)
    lp_daily['month'] = lp_daily.index.month_name()
    return lp_daily

@st.cache_data
def load_weekly_data():
    return pd.read_csv('data/preprocessed/lp_weekly.csv', parse_dates=['date'], index_col=0)

@st.cache_data
def load_monthly_data():
    return pd.read_csv('data/preprocessed/lp_monthly.csv', parse_dates=['date'], index_col=0)

@st.cache_data
def load_avg_daily_profile():
    return pd.read_csv('data/preprocessed/avg_lp.csv', parse_dates=['time'])


def preprocess_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=['datetime'])

    # Daily
    df_daily = df.resample('D', on='datetime').sum().reset_index()
    df_daily['month'] = df_daily['datetime'].dt.month_name()

    # Weekly
    df_weekly = df.resample('W-Mon', on='datetime').sum().reset_index()
    df_weekly['year_week'] = df_weekly['datetime'].dt.strftime('%Y-%U')
    df_weekly['week_number'] = df_weekly['datetime'].dt.week

    # Monthly
    df_monthly = df.resample('M', on='datetime').sum().reset_index()
    df_monthly['month-year'] = df_monthly['datetime'].dt.strftime('%Y-%m')
    df_monthly['month'] = df_monthly['datetime'].dt.month_name()

    # AverageDailyProfile
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.day_name()
    avg_daily_profile = df.groupby(['weekday', 'hour']).mean().reset_index()

    return df_daily, df_weekly, df_monthly, avg_daily_profile


def plot_daily(lp_daily, order):
    if order == "Descending":
        lp_daily = lp_daily.sort_values(by="date", ascending=False)
    fig = px.bar(lp_daily.reset_index(), x='date', y='consumption_kwh',
                 title='Daily Electricity Consumption (kWh)',
                 labels={'consumption_kwh':'kWh', 'date': 'Date'},
                 color='month',
                 category_orders={"month": list(pd.date_range(lp_daily.index.min(), lp_daily.index.max(), freq='M').month_name())},
                 template='plotly_dark',
                 width=1500, height=500)

    fig.update_xaxes(tickangle=45, tickvals=pd.date_range(lp_daily.index.min(), lp_daily.index.max(), freq='MS'))
    st.plotly_chart(fig)

def plot_weekly(lp_weekly, order):
    if order == "Descending":
        lp_weekly = lp_weekly.sort_values(by="year_week", ascending=False)

    lp_weekly = lp_weekly.reset_index()

    fig = px.bar(lp_weekly, x='year_week', y='consumption_kwh',
                 title='Weekly Electricity Consumption (kWh)',
                 labels={'consumption_kwh':'kWh', 'year_week': 'Week'},
                 color='week_number',
                 template='plotly_dark',
                 width=1500, height=500)
    fig.update_xaxes(tickvals=lp_weekly['year_week'].unique(), 
                     tickangle=45, type='category')
    st.plotly_chart(fig)

def plot_monthly(lp_monthly, order):
    if order == "Descending":
        lp_monthly = lp_monthly.sort_values(by="month-year", ascending=False)

    lp_monthly = lp_monthly.reset_index()
    
    fig = px.bar(lp_monthly, x='month-year', y='consumption_kwh', 
             title='Monthly Electricity Consumption (kWh)', 
             labels={'consumption_kwh':'kWh', 'month-year': 'Month'},
             color='month',
             template='plotly_dark',
             width=1500, height=500)

    fig.update_xaxes(tickvals=lp_monthly['month-year'].unique(), 
                 tickangle=45, type='category')
    fig.update_layout(bargap=0.1, barmode='group')

    st.plotly_chart(fig)

def plot_avg_daily_profile(avg_lp, order):
    if order == "Descending":
        avg_lp = avg_lp.sort_values(by="time", ascending=False)
    fig = px.line(avg_lp, x='time', y='consumption_kw', color='weekday',
                  labels={'consumption_kwh': 'kWh', 'time': 'Hour of the Day'},
                  title='Average Daily Profile per Day of the Week',
                  template='plotly_dark',
                  width=1500, height=500)
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    
def main():
    st.title("Energy Consumption Analysis")
    st.sidebar.title("Options")

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    order = st.sidebar.radio("Order data by:", ["Ascending", "Descending"])
    plot_type = st.sidebar.radio("Choose the analysis type:", ["None", "Daily", "Weekly", "Monthly", "Average Daily Load Profile"], key='plot_type')

    if plot_type == "None":

        st.write("Welcome to the Energy Consumption Analysis App!")
        st.write("Please upload a CSV file or use the preloaded LP and select an analysis type to get started.")
        st.image("Landing_page.png")
        
    elif uploaded_file:
        data_daily, data_weekly, data_monthly, avg_daily_profile = preprocess_uploaded_data(uploaded_file)

        if plot_type == "Daily":
            plot_daily(data_daily, order)
        elif plot_type == "Weekly":
            plot_weekly(data_weekly, order)
        elif plot_type == "Monthly":
            plot_monthly(data_monthly, order)
        elif plot_type == "Average Daily Load Profile":
            plot_avg_daily_profile(avg_daily_profile, order)
    elif plot_type == "Daily":
        data = load_daily_data()
        plot_daily(data, order)
    elif plot_type == "Weekly":
        data = load_weekly_data()
        plot_weekly(data, order)
    elif plot_type == "Monthly":
        data = load_monthly_data()
        plot_monthly(data, order)
    elif plot_type == "Average Daily Load Profile":
        data = load_avg_daily_profile()
        plot_avg_daily_profile(data, order)


if __name__ == "__main__":
    main()