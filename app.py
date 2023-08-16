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
    
    lp_weekly = lp_weekly.reset_index()

    fig = px.bar(lp_weekly, x='year_week', y='consumption_kwh',
                 title='Weekly Electricity Consumption (kWh)',
                 labels={'consumption_kwh':'kWh', 'year_week': 'Week'},
                 color='week_number',
                 template='plotly_dark',
                 width=1500, height=500)

    fig.update_xaxes(tickvals=lp_weekly['year_week'].unique(), 
                     tickangle=45, type='category')  # Setting x-axis type to category
    st.plotly_chart(fig)


def plot_monthly(lp_monthly):
    
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

def plot_avg_daily_profile(avg_lp):
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

    plot_type = st.sidebar.radio("Choose the analysis type:", ["Daily", "Weekly", "Monthly", "Average Daily Load Profile"], key='plot_type')
    if plot_type == "Daily":
        data = load_daily_data()
        plot_daily(data)
    elif plot_type == "Weekly":
        data = load_weekly_data()
        data.drop(['month-year'], inplace=True, axis=1)
        st.write('Selected Weekly')
        plot_weekly(data)
    elif plot_type == "Monthly":
        data = load_monthly_data()
        plot_monthly(data)
    elif plot_type == "Average Daily Load Profile":
        data = load_avg_daily_profile()
        plot_avg_daily_profile(data)    

if __name__ == "__main__":
    main()
