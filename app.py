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


def plot_daily(lp_daily):
    fig = px.bar(lp_daily.reset_index(), x='date', y='consumption_kwh',
                 title='Daily Electricity Consumption (kWh)',
                 labels={'consumption_kwh':'kWh', 'date': 'Date'},
                 color='month',
                 category_orders={"month": list(pd.date_range(lp_daily.index.min(), lp_daily.index.max(), freq='M').month_name())},
                 template='plotly_dark',
                 width=1500, height=500)

    fig.update_xaxes(tickangle=45, tickvals=pd.date_range(lp_daily.index.min(), lp_daily.index.max(), freq='MS'))
    st.plotly_chart(fig)
    
def plot_weekly(lp_weekly):
    fig = px.bar(lp_weekly.reset_index(), x='year_week', y='consumption_kwh',
                 title='Weekly Electricity Consumption (kWh)',
                 labels={'consumption_kwh':'kWh', 'year_week': 'Week'},
                 color='week_number',
                 template='plotly_dark',
                 width=1500, height=500)

    fig.update_xaxes(tickvals=lp_weekly['year_week'].unique(), tickangle=45)
    st.plotly_chart(fig)

def plot_monthly(lp_monthly):
    
    fig = px.bar(lp_monthly, x='month-year', y='consumption_kwh', 
             title='Monthly Electricity Consumption (kWh)', 
             labels={'consumption_kwh':'kWh', 'month-year': 'Month'},
             color_discrete_sequence=['#f63366'],
             template='plotly_white',
             width=1500, height=500)

    fig.update_xaxes(tickvals=lp_monthly['month-year'].unique(), 
                 tickangle=45)

    st.plotly_chart(fig)

def main():
    st.title("Energy Consumption Analysis")
    st.sidebar.title("Options")

    plot_type = st.sidebar.radio("Choose the analysis type:", ["Daily", "Weekly", "Monthly"], key='plot_type')
    if plot_type == "Daily":
        data = load_daily_data()
        plot_daily(data)
    elif plot_type == "Weekly":
        data = load_weekly_data()
        st.write('Selected Weekly')
        st.write(data.head())
        st.write(data.describe())
        plot_weekly(data)
    elif plot_type == "Monthly":
        data = load_monthly_data()
        plot_monthly(data)

if __name__ == "__main__":
    main()
