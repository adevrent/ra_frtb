
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Notebook-derived imports
import math
import pyreadr
import warnings
import lifelines
import panel as pn
import statsmodels.api as sm
from datetime import timedelta
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.charts import *
from pyecharts.components import Table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter, WeibullAFTFitter, CoxPHFitter
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, PiecewiseExponentialFitter, NelsonAalenFitter, SplineFitter
from lifelines.utils import find_best_parametric_model, median_survival_times
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from scipy.stats import norm
from timeit import default_timer as timer
from itertools import combinations, batched
from rich.progress import track, BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from IPython.display import display, HTML, Markdown
from rich.progress import track
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn



import inspect

def call_with_supported_kwargs(fn, *args, **kwargs):
    # Calls function dropping any kwargs it doesn't accept
    sig = inspect.signature(fn)
    params = sig.parameters
    supported = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **supported)

def _filter_by_date_range(df, start_iso, end_iso):
    if df is None:
        return df
    start = pd.to_datetime(start_iso)
    end = pd.to_datetime(end_iso)
    candidates = [c for c in df.columns if c.lower() == "date"]
    col = None
    if candidates:
        col = candidates[0]
    else:
        for c in df.columns:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[c] = parsed
                    col = c
                    break
            except Exception:
                continue
    if col is None:
        return df
    df[col] = pd.to_datetime(df[col])
    m = (df[col] >= start) & (df[col] <= end)
    return df.loc[m].copy()

# ==== Extracted function/class code from notebooks ====

# ---- from notebook ----
def next_business_day(date, holidays_df):
    holidays = set(holidays_df.iloc[:,0])
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:
        next_day += timedelta(days=1)
    return next_day
    
def classify_time_bucket(months):
    if 0 <= months <= 1:
        return "0-1M"
    elif 1 < months <= 3:
        return "1-3M"
    elif 3 < months <= 6:
        return "3-6M"
    elif 6 < months <= 12:
        return "6-12M"    
    elif 12 < months <= 24:
        return "1-2Y"
    elif 24 < months <= 36:
        return "2-3Y"
    elif 36 < months <= 48:
        return "3-4Y"
    elif 48 < months <= 60:
        return "4-5Y"
    elif 60 < months <= 120:
        return "5-10Y"
    elif months > 120:
        return "+10Y"
    else:
        return "--"

def rolling_dates(data, year = 5):
    rolling_dates = []
    period = year*12
    for i in range(len(data) - (period-1)):
        START_DATE = data.loc[i, "first"]
        END_DATE = data.loc[i + (period-1), "last"]
        rolling_dates.append({"START_DATE": START_DATE, "END_DATE": END_DATE})
    
    rolling_dates = pd.DataFrame(rolling_dates)
    
    return rolling_dates

def data_model(data, data_year_basis = None, start_date = None, end_date = None, excel = False):
    
    data['REPORT_DATE'] = pd.to_datetime(data['REPORT_DATE'])

    if data_year_basis is not None:
        df_rolling_dates = rolling_dates(Monthly_Business_Days, year = data_year_basis)
        start_date = df_rolling_dates['START_DATE'].iloc[-1]
        end_date = df_rolling_dates['END_DATE'].iloc[-1]
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)]
        data.reset_index(drop=True, inplace=True)

    if start_date is not None and  end_date is not None:
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)]
        data.reset_index(drop=True, inplace=True)
        
    return data 

def df_core_deposit(data, 
                    branch = None, 
                    product = None, 
                    time_bucket = None,
                    currency = None):
   
    # Branch,Product_Code,Time_Bucket,Currency
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Product_Code,Time_Bucket,Currency	
    if branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Time_Bucket,Currency
    if branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Product_Code,Currency
    if branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Time_Bucket,Currency	
    if branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Product_Code,Currency
    if branch is None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Currency
    if branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Currency
    if branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))                  
        
    return data

def core_deposit_analysis(data, 
                          branch = None, 
                          product = None, 
                          time_bucket = None, 
                          currency = None,
                          freq = None, type = None, nsim = 100000, excel = False, plot = False):
    
    wd_SAVE = 'D:/'

    # Branch,Product_Code,Time_Bucket,Currency
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Product_Code,Time_Bucket,Currency	
    if branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Time_Bucket,Currency
    if branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Product_Code,Currency
    if branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Time_Bucket,Currency	
    if branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Product_Code,Currency
    if branch is None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))
            
    # Branch,Currency
    if branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Currency
    if branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))                      

    if data.empty:
        raise ValueError("The filtered data is empty. Please check the filter criteria.")
    # print(data)
    data = data.set_index("Report_Date")
    
    if freq == 'W':
        if type == 'mean':
            data = data.resample('W').mean()
        elif type == 'roll':
            data['Notional'] = data['Notional'].rolling(window=5).mean() 
            data.dropna(inplace=True)        
        else:
            data = data.resample('W').last()        
    elif freq == 'M':
        if type == 'mean':
            data = data.resample('M').mean()
        elif type == 'roll':
            data['Notional'] = data['Notional'].rolling(window=21).mean() 
            data.dropna(inplace=True)        
        else:
            data = data.resample('M').last()           
    elif freq == 'Y':
        if type == 'mean':
            data = data.resample('Y').mean()        
        else:
            data = data.resample('Y').last() 
    else:
        data

    period = pd.to_datetime(data.index)
    df = data['Notional'].values.reshape(-1, 1)
    # print(df)

    min_month_year = data.index.min().strftime('%b-%Y')
    max_month_year = data.index.max().strftime('%b-%Y')
    
    df_log = np.log(df)
    
    
    if freq == 'Y':        
        hpfilter_lamb = 100
    elif freq == 'M':
        hpfilter_lamb = 14400
    elif freq == 'W':
        if type == 'mean':
            hpfilter_lamb = 125000
        elif type == 'roll':
            hpfilter_lamb = 129600
        else:
            hpfilter_lamb = 125000        
    elif freq == 'D':
        hpfilter_lamb = 129600    
    else:
        hpfilter_lamb = 129600
        
    cycle, trend = sm.tsa.filters.hpfilter(df_log, lamb = hpfilter_lamb) # Annually=100 / Quarterly=1600 / Monthly=14400 / Weekly=125000 / Daily=129600 
    
    df_sigma = np.std(df_log[:, 0] - trend)
    
    df_log_core = trend - norm.ppf(0.99, loc=0, scale=1) * df_sigma
    
    df_core = np.exp(df_log_core)
    df_core_percentage = df_core[-1] / np.exp(trend[-1])
    df_sigma = df_sigma * (1 - df_core_percentage)

    df_log_MPA = np.zeros((nsim + 1, 1))
    df_log_MPA[0] = df_log_core[-1]
    
    for j in range(1, nsim + 1):
        df_log_MPA[j] = df_log_core[-1] - 0.5 * df_sigma**2 * j - df_sigma * np.sqrt(j) * norm.ppf(0.99)

    df_MPA = np.exp(df_log_MPA)
    
    df_MPCF = np.zeros((nsim))
    
    for j in range(nsim):
        df_MPCF[j] = df_MPA[j] - df_MPA[j + 1]
    
    width = (df_MPA[:-1] + df_MPA[1:]) / 2
    mean_life = np.sum(width) / df_MPA[0]
    horizon = int(round(mean_life.item()))

    df_MPCF_adj = np.zeros((horizon))
    
    for j in range(horizon):
        df_MPCF_adj[j] = df_MPCF[j] + df_MPA[horizon] / horizon
    
    df_MPA_adj = np.zeros((horizon + 1))
    df_MPA_adj[0] = df_MPA[0]
    
    for j in range(horizon):
        df_MPA_adj[j + 1] = df_MPA_adj[j] - df_MPCF_adj[j]
    
    persistence = df_MPA_adj / df_MPA_adj[0]

    persistence2 = np.ones(horizon + 1)
    
    for j in range(1, horizon + 1):
        persistence2[j] = persistence[j - 1] * df_core_percentage
    
    final_result = np.zeros((horizon + 1))
    final_result[:] = persistence2
    
    distribution = persistence2[:-1] - persistence2[1:]
    average_life = np.dot(distribution, np.arange(len(distribution)))
    horizon_life = horizon
    core_amount = df_core[-1]

    digit = 2
    
    if freq == 'D':
        mean_days = round(average_life,0)
        mean_months = round(average_life/21,digit)
        mean_years = round(average_life/252,digit)
        
        horizon_days = round(horizon_life,0)
        horizon_months = round(horizon_life/21,digit)
        horizon_years = round(horizon_life/252,digit)   

    if freq == 'W':
        if type == 'mean':
            mean_days = round(average_life*5/1,0)
            mean_months = round(average_life*5/21,digit)
            mean_years = round(average_life*5/252,digit)
            
            horizon_days = round(horizon_life*5/1,0)
            horizon_months = round(horizon_life*5/21,digit)
            horizon_years = round(horizon_life*5/252,digit)
        elif type == 'roll':
            mean_days = round(average_life,0)
            mean_months = round(average_life/21,digit)
            mean_years = round(average_life/252,digit)
            
            horizon_days = round(horizon_life,0)
            horizon_months = round(horizon_life/21,digit)
            horizon_years = round(horizon_life/252,digit)
        else:
            mean_days = round(average_life*5/1,0)
            mean_months = round(average_life*5/21,digit)
            mean_years = round(average_life*5/252,digit)
            
            horizon_days = round(horizon_life*5/1,0)
            horizon_months = round(horizon_life*5/21,digit)
            horizon_years = round(horizon_life*5/252,digit)  
            
    if freq == 'M':
        mean_days = round(average_life*21,0)
        mean_months = round(average_life,digit)
        mean_years = round(average_life/12,digit)      
        
        horizon_days = round(horizon_life*21,0)
        horizon_months = round(horizon_life,digit)
        horizon_years = round(horizon_life/12,digit)

    if freq == 'Y':
        mean_days = round(average_life/252,0)
        mean_months = round(average_life/12,digit)
        mean_years = round(average_life,digit)
        
        horizon_days = round(horizon_life/252,0)
        horizon_months = round(horizon_life/12,digit)
        horizon_years = round(horizon_life,digit)
    
    if excel:
        output_df = pd.DataFrame({
            'REPORT_DATE': period,
            'DEMAND_DEPOSITS': df.flatten(),
            'TREND': np.exp(trend).flatten(),
            'CORE_DEPOSITS': df_core.flatten()
        })        
        output_df.to_excel(wd_SAVE+'Core_Deposit_Analysis.xlsx', index=False)
        
    if plot:
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Notional / Core Deposit", "Persistence"))

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Notional'],
                mode='lines',
                line=dict(color='darkorange'),
                name='Notional / Core Deposit'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=df_core,
                mode='lines',
                fill='tozeroy',
                line=dict(color='royalblue')                
            ),
            row=1, col=1
        )
        
        y_min = min(data['Notional'].min(), df_core.min())
        y_max = max(data['Notional'].max(), df_core.max())

        fig.add_trace(
            go.Scatter(
                x=np.arange(horizon_days + 1), 
                y=persistence,
                mode='lines',
                fill='tozeroy',
                line=dict(color='darkorange'),
                name='Persistence'
            ),            
            row=1, col=2
        )
               
        #fig.add_hline(y=0.5, line_width=2, line_dash="dot", line_color="royalblue", row=1, col=2)
        fig.add_vline(x=mean_days, line_width=2, line_dash="dot", line_color="royalblue", row=1, col=2)
        
        y_min = min(data['Notional'].min(), df_core.min())
        y_max = max(data['Notional'].max(), df_core.max())

        fig.update_layout(
            title="Core Deposit Analysis ("+ min_month_year + ' - ' + max_month_year + ')',
            width=1000,
            height=500,
            plot_bgcolor="#fff",
            showlegend=False,
            yaxis=dict(range=[0, y_max])  # First y-axis
            #yaxis2=dict(range=[0, y_max])  # Second y-axis
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        #fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Demand Deposit", row=1, col=1)
        #fig.update_yaxes(title_text="Demand Deposit", row=1, col=2)
        
        fig.show()

    table_a = {'Core Deposit %': [],
               'Core Deposits (Mio)': []}

    table_b = {'Mean Life (Days)': [],
               'Mean Life (Months)': [],
               'Mean Life (Years)': []}

    table_c = {'Horizon Life (Days)': [],
               'Horizon Life (Months)': [],
               'Horizon Life (Years)': []}
    
    table_a['Core Deposit %'].append(round(df_core_percentage.item()*100,2))
    table_a['Core Deposits (Mio)'].append(round(core_amount.item()/1000000,0))   
    
    table_b['Mean Life (Days)'].append(mean_days.item())
    table_b['Mean Life (Months)'].append(mean_months.item())
    table_b['Mean Life (Years)'].append(mean_years.item())

    table_c['Horizon Life (Days)'].append(horizon_days)
    table_c['Horizon Life (Months)'].append(horizon_months)
    table_c['Horizon Life (Years)'].append(horizon_years)
    
    table_a = pd.DataFrame(table_a)
    table_b = pd.DataFrame(table_b)
    table_c = pd.DataFrame(table_c)
    
    display(HTML(table_a.to_html(index=False)))
    display(HTML(table_b.to_html(index=False)))
    display(HTML(table_c.to_html(index=False)))
    
    return 

def core_deposit_analysis_rolling(data, 
                                  branch = None, 
                                  product = None, 
                                  time_bucket = None, 
                                  currency = None,
                                  freq = None, type = None, nsim = 100000, excel = False, plot = False):
    
    wd_SAVE = 'D:/'

    # Branch,Product_Code,Time_Bucket,Currency
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Product_Code,Time_Bucket,Currency	
    if branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Time_Bucket,Currency
    if branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Product_Code,Currency
    if branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Time_Bucket,Currency	
    if branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Product_Code,Currency
    if branch is None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Branch,Currency
    if branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))    
    # Currency
    if branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
        data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))                      

    data = data.set_index("Report_Date")
    
    if freq == 'W':
        if type == 'mean':
            data = data.resample('W').mean()
        elif type == 'roll':
            data['Notional'] = df_md['Notional'].rolling(window=5).mean() 
            data.dropna(inplace=True)        
        else:
            data = data.resample('W').last()        
    elif freq == 'M':
        if type == 'mean':
            data = data.resample('M').mean()
        elif type == 'roll':
            data['Notional'] = df_md['Notional'].rolling(window=21).mean() 
            data.dropna(inplace=True)        
        else:
            data = data.resample('M').last()           
    elif freq == 'Y':
        if type == 'mean':
            data = data.resample('Y').mean()        
        else:
            data = data.resample('Y').last() 
    else:
        data

    period = pd.to_datetime(data.index)
    df = data['Notional'].values.reshape(-1, 1)

    min_month_year = data.index.min().strftime('%b-%Y')
    max_month_year = data.index.max().strftime('%b-%Y')
    
    df_log = np.log(df)
    
    if freq == 'Y':        
        hpfilter_lamb = 100
    elif freq == 'M':
        hpfilter_lamb = 14400
    elif freq == 'W':
        if type == 'mean':
            hpfilter_lamb = 125000
        elif type == 'roll':
            hpfilter_lamb = 129600
        else:
            hpfilter_lamb = 125000        
    elif freq == 'D':
        hpfilter_lamb = 129600    
    else:
        hpfilter_lamb = 129600
        
    cycle, trend = sm.tsa.filters.hpfilter(df_log, lamb = hpfilter_lamb) # Annually=100 / Quarterly=1600 / Monthly=14400 / Weekly=125000 / Daily=129600 
    
    df_sigma = np.std(df_log[:, 0] - trend)
    
    df_log_core = trend - norm.ppf(0.99, loc=0, scale=1) * df_sigma
    
    df_core = np.exp(df_log_core)
    df_core_percentage = df_core[-1] / np.exp(trend[-1])
    df_sigma = df_sigma * (1 - df_core_percentage)

    df_log_MPA = np.zeros((nsim + 1, 1))
    df_log_MPA[0] = df_log_core[-1]
    
    for j in range(1, nsim + 1):
        df_log_MPA[j] = df_log_core[-1] - 0.5 * df_sigma**2 * j - df_sigma * np.sqrt(j) * norm.ppf(0.99)

    df_MPA = np.exp(df_log_MPA)
    
    df_MPCF = np.zeros((nsim))
    
    for j in range(nsim):
        df_MPCF[j] = df_MPA[j] - df_MPA[j + 1]
    
    width = (df_MPA[:-1] + df_MPA[1:]) / 2
    mean_life = np.sum(width) / df_MPA[0]
    horizon = int(round(mean_life.item()))

    df_MPCF_adj = np.zeros((horizon))
    
    for j in range(horizon):
        df_MPCF_adj[j] = df_MPCF[j] + df_MPA[horizon] / horizon
    
    df_MPA_adj = np.zeros((horizon + 1))
    df_MPA_adj[0] = df_MPA[0]
    
    for j in range(horizon):
        df_MPA_adj[j + 1] = df_MPA_adj[j] - df_MPCF_adj[j]
    
    persistence = df_MPA_adj / df_MPA_adj[0]

    persistence2 = np.ones(horizon + 1)
    
    for j in range(1, horizon + 1):
        persistence2[j] = persistence[j - 1] * df_core_percentage
    
    final_result = np.zeros((horizon + 1))
    final_result[:] = persistence2
    
    distribution = persistence2[:-1] - persistence2[1:]
    average_life = np.dot(distribution, np.arange(len(distribution)))
    horizon_life = horizon
    core_amount = df_core[-1]

    digit = 2
    
    if freq == 'D':
        mean_days = round(average_life,0)
        mean_months = round(average_life/21,digit)
        mean_years = round(average_life/252,digit)
        
        horizon_days = round(horizon_life,0)
        horizon_months = round(horizon_life/21,digit)
        horizon_years = round(horizon_life/252,digit)   

    if freq == 'W':
        if type == 'mean':
            mean_days = round(average_life*5/1,0)
            mean_months = round(average_life*5/21,digit)
            mean_years = round(average_life*5/252,digit)
            
            horizon_days = round(horizon_life*5/1,0)
            horizon_months = round(horizon_life*5/21,digit)
            horizon_years = round(horizon_life*5/252,digit)
        elif type == 'roll':
            mean_days = round(average_life,0)
            mean_months = round(average_life/21,digit)
            mean_years = round(average_life/252,digit)
            
            horizon_days = round(horizon_life,0)
            horizon_months = round(horizon_life/21,digit)
            horizon_years = round(horizon_life/252,digit)
        else:
            mean_days = round(average_life*5/1,0)
            mean_months = round(average_life*5/21,digit)
            mean_years = round(average_life*5/252,digit)
            
            horizon_days = round(horizon_life*5/1,0)
            horizon_months = round(horizon_life*5/21,digit)
            horizon_years = round(horizon_life*5/252,digit)  
            
    if freq == 'M':
        mean_days = round(average_life*21,0)
        mean_months = round(average_life,digit)
        mean_years = round(average_life/12,digit)      
        
        horizon_days = round(horizon_life*21,0)
        horizon_months = round(horizon_life,digit)
        horizon_years = round(horizon_life/12,digit)

    if freq == 'Y':
        mean_days = round(average_life/252,0)
        mean_months = round(average_life/12,digit)
        mean_years = round(average_life,digit)
        
        horizon_days = round(horizon_life/252,0)
        horizon_months = round(horizon_life/12,digit)
        horizon_years = round(horizon_life,digit)
        
    return df_core_percentage, core_amount, mean_days, mean_years, horizon_days, horizon_years

# ---- from notebook ----
Turkey_Holidays = pd.read_excel(f"{wd_PRAM}/Turkey_Holidays.xlsx")
Turkey_Holidays['TURKEY_HOLIDAYS'] = pd.to_datetime(Turkey_Holidays['TURKEY_HOLIDAYS'])

Business_Days = pd.date_range(start="2017-01-01", end="2025-12-31", freq='B')
Turkey_Business_Days = Business_Days[~Business_Days.isin(Turkey_Holidays['TURKEY_HOLIDAYS'])]

Monthly_Business_Days = (
    pd.DataFrame({"Business_Days": Turkey_Business_Days})
    .assign(YearMonth=lambda df: df["Business_Days"].dt.to_period("M"))
    .groupby("YearMonth")["Business_Days"]
    .agg(["first", "last"])
    .reset_index()
)

def next_business_day(date, holidays):
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:
        next_day += timedelta(days=1)
    return next_day

def classify_time_bucket(months):
    if 0 <= months <= 6:
        return "0-6M"
    elif 6 < months <= 12:
        return "6-12M"
    elif 12 < months <= 24:
        return "1-2Y"
    elif 24 < months <= 36:
        return "2-3Y"
    elif 36 < months <= 48:
        return "3-4Y"
    elif 48 < months <= 60:
        return "4-5Y"
    elif 60 < months <= 120:
        return "5-10Y"
    elif months > 120:
        return "+10Y"
    else:
        return "--"

def rolling_dates(data, year = 5):
    rolling_dates = []
    period = year*12
    for i in range(len(data) - (period-1)):
        START_DATE = data.loc[i, "first"]
        END_DATE = data.loc[i + (period-1), "last"]
        rolling_dates.append({"START_DATE": START_DATE, "END_DATE": END_DATE})
    
    rolling_dates = pd.DataFrame(rolling_dates)
    
    return rolling_dates

def separate_customer_id(customer_id):
    try:
        parts = customer_id.split(' // ')
        return pd.Series(parts, index=['BRANCH', 'CUSTOMER_NO', 'PRODUCT_CODE', 'CURRENCY'])
    except:
        return pd.Series([None, None, None, None], index=['BRANCH', 'CUSTOMER_NO', 'PRODUCT_CODE', 'CURRENCY'])

def move_string_to_end(s):
    parts = s.split(',')
    if "Currency" in parts:
        parts.remove("Currency")
        parts.append("Currency")
    return ",".join(parts)  

def kaplan_meier_estimator_custom(df):

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    kme = kaplan_meier_estimator(df["Status_TF"], df["Survival_in_Days"], conf_type="log-log")
    
    time_points = kme[0]
    survival_probabilities = kme[1]
    confidence_interval = kme[2]
    
    median_survival_time = np.interp(0.5, survival_probabilities[::-1], time_points[::-1])
    
    survival_probabilities_diff = survival_probabilities - 0.5
    median_survival_time_adj = np.interp(min(survival_probabilities_diff), survival_probabilities_diff[::-1], time_points[::-1])
    
    if min(survival_probabilities) > 0.5:  
        median_survival_time = median_survival_time_adj
        
    return time_points, survival_probabilities, confidence_interval, median_survival_time

def move_string_to_end(s):
    parts = s.split(',')
    if "Currency" in parts:
        parts.remove("Currency")
        parts.append("Currency")
    return ",".join(parts)

def data_model(data, data_year_basis = None, start_date = None, end_date = None, excel = False, Max_Report_Date = None):
    if Max_Report_Date is None:
        Max_Report_Date = data.groupby('CUSTOMER_ID')['REPORT_DATE'].max().reset_index().rename(columns={'REPORT_DATE': 'MAX_RPT'})
    data['REPORT_DATE'] = pd.to_datetime(data['REPORT_DATE'])

    if data_year_basis is not None:
        df_rolling_dates = rolling_dates(Monthly_Business_Days, year = data_year_basis)
        start_date = df_rolling_dates['START_DATE'].iloc[-1]
        end_date = df_rolling_dates['END_DATE'].iloc[-1]
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)]
        data.reset_index(drop=True, inplace=True)

    if start_date is not None and  end_date is not None:
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)]
        data.reset_index(drop=True, inplace=True)
        
    currency_series = data["CURRENCY"]
    
    data = (
        data
        .assign(
            NO=lambda x: np.where(
                x['REPORT_DATE'] == x['CUSTOMER_ID'].map(Max_Report_Date.set_index('CUSTOMER_ID')['MAX_RPT']), 1, 0
            )
        )
        .query("NO == 1")
        .assign(LAST_REPORT_DATE=lambda x: x['REPORT_DATE'].max())
        .assign(MAX_RPT_DATE=lambda x: x['CUSTOMER_ID'].map(Max_Report_Date.set_index('CUSTOMER_ID')['MAX_RPT']))
        .groupby('CUSTOMER_ID')
        .agg({
            'REPORT_DATE': 'max',
            'EFFECTIVE_DATE': 'max',
            'MATURITY': 'max',
            'NOTIONAL': 'max',
            'LAST_REPORT_DATE': 'max'
        })
        .reset_index()
        .assign(
            MATURITY_CONT=lambda x: np.where(x['MATURITY'].dt.weekday >= 5, 1, 0),
            MATURITY_NEW=lambda x: np.where(
                x['MATURITY'].dt.weekday == 5, x['MATURITY'] + timedelta(days=2),
                np.where(x['MATURITY'].dt.weekday == 6, x['MATURITY'] + timedelta(days=1), x['MATURITY'])
            ),
            NEXT_BUS_DATE=lambda x: x['REPORT_DATE'].apply(lambda x: next_business_day(x, Turkey_Holidays)),        
            Maturity_Periods_D=lambda x: (x['MATURITY_NEW'] - x['EFFECTIVE_DATE']).dt.days,
            Maturity_Periods_M=lambda x: (x['MATURITY_NEW'] - x['EFFECTIVE_DATE']).dt.days / (365.25 / 12),
            SURV_TIME=lambda x: (x['NEXT_BUS_DATE'] - x['EFFECTIVE_DATE']).dt.days,
            DIFF_TIME=lambda x: (x['MATURITY_NEW'] - x['NEXT_BUS_DATE']).dt.days,
            #DIFF_MONTH=lambda x: (x['MATURITY_NEW'].apply(lambda date: date - (date - pd.offsets.MonthEnd(1)))).dt.days,
            DIFF_MONTH=lambda x: (x['MATURITY_NEW'] - (x['MATURITY_NEW'] - pd.DateOffset(months=1))).dt.days,
            TIME_BUCKET=lambda x: x['Maturity_Periods_M'].apply(classify_time_bucket)        
        )
        .assign(
            Prepayment_Status=lambda x: np.where(
                (x['NEXT_BUS_DATE'] < x['MATURITY_NEW']) & (x['DIFF_TIME'] > x['DIFF_MONTH']),
                "Early", "OnTime"
            )
        )    
    )
    # print(data.head())
    data["CURRENCY"] = currency_series.loc[data.index].values

    data = data[data['DIFF_TIME'] >= 0]
    data[['BRANCH', 'CUSTOMER_NO','CUSTOMER_EX_NO', 'PRODUCT_CODE']] = (data['CUSTOMER_ID'].str.split('//', expand=True))   
    # print(data.head())  # TODO
    selected_columns = [
        'REPORT_DATE','CUSTOMER_ID','BRANCH','CUSTOMER_NO','CUSTOMER_EX_NO','PRODUCT_CODE',
        'CURRENCY','EFFECTIVE_DATE','MATURITY','NOTIONAL',
        'LAST_REPORT_DATE','MATURITY_CONT','MATURITY_NEW','NEXT_BUS_DATE',
        'Maturity_Periods_D','Maturity_Periods_M','SURV_TIME',
        'DIFF_TIME','DIFF_MONTH','TIME_BUCKET','Prepayment_Status'
    ]
    data = data[selected_columns]
  
    data = data[selected_columns]
    data = data[['REPORT_DATE','NOTIONAL','BRANCH','CURRENCY' ,'PRODUCT_CODE','TIME_BUCKET','SURV_TIME','Prepayment_Status']].copy()
    data.rename(columns={'REPORT_DATE': 'Report_Date', 'NOTIONAL': 'Notional','BRANCH': 'Branch', 'CURRENCY': 'Currency', 'PRODUCT_CODE': 'Product_Code',
                          'TIME_BUCKET': 'Time_Bucket', 'SURV_TIME': 'Survival_in_Days', 'Prepayment_Status': 'Status'}, inplace=True)    
    data['Status_TF'] = data['Status']
    data['Status_YN'] = data['Status']
    data['Status'] = data['Status'].replace({'Early': 1, 'OnTime': 0})
    data['Status_TF'] = data['Status_TF'].replace({'Early': True, 'OnTime': False})
    data['Status_YN'] = data['Status_YN'].replace({'Early': 'Yes', 'OnTime': 'No'})
    data["Prepayment_Notional"] = data["Notional"] * (data["Status"] == 1)
        
    if excel:
        data = pd.DataFrame(data)    
        data.to_excel(wd_SAVE+'Report_Prepayment_df.xlsx', index=False)
        
    return data

# ---- from notebook ----
def df_prepayment(data, 
                  branch = None, 
                  product = None, 
                  time_bucket = None,
                  currency = None):
    
    # Branch,Product_Code,Time_Bucket,Currency
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)        
    # Product_Code,Time_Bucket,Currency	
    if branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Branch,Time_Bucket,Currency
    if branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Branch,Product_Code,Currency
    if branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Time_Bucket,Currency	
    if branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Branch,Currency
    if branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Currency
    if branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    return data

# ---- from notebook ----
def next_business_day(date, holidays_df):
    holidays = set(holidays_df.iloc[:,0])
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:
        next_day += timedelta(days=1)
    return next_day
    
def classify_time_bucket(months):
    if 0 <= months <= 1:
        return "0-1M"
    elif 1 < months <= 3:
        return "1-3M"
    elif 3 < months <= 6:
        return "3-6M"
    elif 6 < months <= 12:
        return "6-12M"    
    elif 12 < months <= 24:
        return "1-2Y"
    elif 24 < months <= 36:
        return "2-3Y"
    elif 36 < months <= 48:
        return "3-4Y"
    elif 48 < months <= 60:
        return "4-5Y"
    elif 60 < months <= 120:
        return "5-10Y"
    elif months > 120:
        return "+10Y"
    else:
        return "--"

def rolling_dates(data, year = 5):
    rolling_dates = []
    period = year*12
    for i in range(len(data) - (period-1)):
        START_DATE = data.loc[i, "first"]
        END_DATE = data.loc[i + (period-1), "last"]
        rolling_dates.append({"START_DATE": START_DATE, "END_DATE": END_DATE})
    
    rolling_dates = pd.DataFrame(rolling_dates)
    
    return rolling_dates
    
def separate_customer_id(customer_id):
    try:
        parts = customer_id.split(' // ')
        return pd.Series(parts, index=['BRANCH', 'CUSTOMER_NO', 'PRODUCT_CODE', 'CURRENCY'])
    except:
        return pd.Series([None, None, None, None], index=['BRANCH', 'CUSTOMER_NO', 'PRODUCT_CODE', 'CURRENCY'])

def move_string_to_end(s):
    parts = s.split(',')
    if "Currency" in parts:
        parts.remove("Currency")
        parts.append("Currency")
    return ",".join(parts)
    
def kaplan_meier_estimator_custom(df):

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    kme = kaplan_meier_estimator(df["Status_TF"], df["Survival_in_Days"], conf_type="log-log")
    
    time_points = kme[0]
    survival_probabilities = kme[1]
    confidence_interval = kme[2]
    
    median_survival_time = np.interp(0.5, survival_probabilities[::-1], time_points[::-1])
    
    survival_probabilities_diff = survival_probabilities - 0.5
    median_survival_time_adj = np.interp(min(survival_probabilities_diff), survival_probabilities_diff[::-1], time_points[::-1])
    
    if min(survival_probabilities) > 0.5:  
        median_survival_time = median_survival_time_adj
        
    return time_points, survival_probabilities, confidence_interval, median_survival_time
    
def data_model(data, data_year_basis = None, start_date = None, end_date = None, excel = False):
    
    data['REPORT_DATE'] = pd.to_datetime(data['REPORT_DATE'])

    if data_year_basis is not None:
        df_rolling_dates = rolling_dates(Monthly_Business_Days, year = data_year_basis)
        start_date = df_rolling_dates['START_DATE'].iloc[-1]
        end_date = df_rolling_dates['END_DATE'].iloc[-1]
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)]
        data.reset_index(drop=True, inplace=True)

    if start_date is not None and  end_date is not None:
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)]
        data.reset_index(drop=True, inplace=True)
    
    currency_series = data["CURRENCY"]
    
    data = (
        data
        .assign(
            NO=lambda x: np.where(
                x['REPORT_DATE'] == x['CUSTOMER_ID'].map(Max_Report_Date.set_index('CUSTOMER_ID')['MAX_RPT']), 1, 0
            )
        )
        .query("NO == 1")
        .assign(LAST_REPORT_DATE=lambda x: x['REPORT_DATE'].max())
        .assign(MAX_RPT_DATE=lambda x: x['CUSTOMER_ID'].map(Max_Report_Date.set_index('CUSTOMER_ID')['MAX_RPT']))
        .groupby('CUSTOMER_ID')
        .agg({
            'REPORT_DATE': 'max',
            'EFFECTIVE_DATE': 'max',
            'MATURITY': 'max',
            'NOTIONAL': 'max',
            'LAST_REPORT_DATE': 'max'
        })
        .reset_index()
        .assign(
            MATURITY_CONT=lambda x: np.where(x['MATURITY'].dt.weekday >= 5, 1, 0),
            MATURITY_NEW=lambda x: np.where(
                x['MATURITY'].dt.weekday == 5, x['MATURITY'] + timedelta(days=2),
                np.where(x['MATURITY'].dt.weekday == 6, x['MATURITY'] + timedelta(days=1), x['MATURITY'])
            ),
            NEXT_BUS_DATE=lambda x: x['REPORT_DATE'].apply(lambda x: next_business_day(x, Turkey_Holidays)),        
            Maturity_Periods_D=lambda x: (x['MATURITY_NEW'] - x['EFFECTIVE_DATE']).dt.days,
            Maturity_Periods_M=lambda x: (x['MATURITY_NEW'] - x['EFFECTIVE_DATE']).dt.days / (365.25 / 12),
            SURV_TIME=lambda x: (x['NEXT_BUS_DATE'] - x['EFFECTIVE_DATE']).dt.days,
            DIFF_TIME=lambda x: (x['MATURITY_NEW'] - x['NEXT_BUS_DATE']).dt.days,
            #DIFF_MONTH=lambda x: (x['MATURITY_NEW'].apply(lambda date: date - (date - pd.offsets.MonthEnd(1)))).dt.days,
            DIFF_MONTH=lambda x: (x['MATURITY_NEW'] - (x['MATURITY_NEW'] - pd.DateOffset(months=1))).dt.days,
            TIME_BUCKET=lambda x: x['Maturity_Periods_M'].apply(classify_time_bucket)        
        )
        .assign(
            Early_Withdrawal_Status=lambda x: np.where(
                (x['NEXT_BUS_DATE'] < x['MATURITY_NEW']) & (x['DIFF_TIME'] > x['DIFF_MONTH']),
                "Early", "OnTime"
            )
        )    
    )

    data["CURRENCY"] = currency_series.loc[data.index].values

    data = data[data['DIFF_TIME'] >= 0]    
    data[['BRANCH', 'CUSTOMER_NO','CUSTOMER_EX_NO', 'PRODUCT_CODE']] = (data['CUSTOMER_ID'].str.split('//', expand=True))    
    selected_columns = ['REPORT_DATE','CUSTOMER_ID','BRANCH', 'CUSTOMER_NO','CUSTOMER_EX_NO', 'PRODUCT_CODE', 'CURRENCY', 'EFFECTIVE_DATE', 
                        'MATURITY', 'NOTIONAL', 'LAST_REPORT_DATE', 'MATURITY_CONT', 'MATURITY_NEW', 'NEXT_BUS_DATE', 'Maturity_Periods_D', 
                        'Maturity_Periods_M','SURV_TIME', 'DIFF_TIME', 'DIFF_MONTH', 'TIME_BUCKET', 'Early_Withdrawal_Status']    
    print(data)
    data = data[selected_columns]
    
    data = data[['REPORT_DATE','BRANCH', 'CURRENCY' ,'PRODUCT_CODE','TIME_BUCKET','NOTIONAL','SURV_TIME','Early_Withdrawal_Status']].copy()
    data.rename(columns={'REPORT_DATE': 'Report_Date','BRANCH': 'Branch', 'CURRENCY': 'Currency', 'PRODUCT_CODE': 'Product_Code',
                         'TIME_BUCKET': 'Time_Bucket','NOTIONAL': 'Notional', 'SURV_TIME': 'Survival_in_Days', 'Early_Withdrawal_Status': 'Status'}, inplace=True)
    data['Status_TF'] = data['Status']
    data['Status'] = data['Status'].replace({'Early': 1, 'OnTime': 0})
    data['Status_TF'] = data['Status_TF'].replace({'Early': True, 'OnTime': False})
    data['Early_Withdrawal_Notional'] = data['Notional'] * (data['Status'] == 1)
    
    if excel:
        data = pd.DataFrame(data)    
        data.to_excel(wd_SAVE+'Report_Early_Withdrawal_df.xlsx', index=False)
        
    return data

# ---- from notebook ----
def df_early_withdrawal(data, 
                        branch = None, 
                        product = None, 
                        time_bucket = None,
                        currency = None):
    
    # Branch,Product_Code,Time_Bucket,Currency
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)        
    # Product_Code,Time_Bucket,Currency	
    if branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Branch,Time_Bucket,Currency
    if branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Branch,Product_Code,Currency
    if branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Time_Bucket,Currency	
    if branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Product_Code,Currency
    if branch is None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)            
    # Branch,Currency
    if branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    # Currency
    if branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
        data.reset_index(drop=True, inplace=True)
    return data

st.set_page_config(page_title="FRTB Analyses", layout="wide")

st.title("FRTB Analyses — Core / Prepayment / Early Withdrawal")
st.caption("Upload-only app: no static filepaths. Drop your Turkey_Holidays.xlsx and the corresponding data file for each analysis.")

analysis = st.radio(
    "Select analysis",
    ["Core Analysis", "Prepayment Analysis", "Early Withdrawal Analysis"],
    horizontal=True
)

if analysis == "Core Analysis":
    st.subheader("Inputs")
    date_start = st.text_input("date_start (ISO, e.g. 2017-01-01)", value="2017-01-01")
    date_end = st.text_input("date_end (ISO, e.g. 2018-01-01)", value="2018-01-01")
    currency = st.selectbox("currency", ["USD", "EUR", "TRY"], index=2)
    freq = st.selectbox("freq", ["W", "M", "Y"], index=1)
    agg_type = st.selectbox("type", ["mean", "roll"], index=0)

    holidays_file = st.file_uploader("Upload Turkey_Holidays.xlsx", type=["xlsx"], key="hol_core")
    data_file = st.file_uploader("Upload CORE_DEPOSIT_DATA.xlsx", type=["xlsx"], key="data_core")

    if holidays_file and data_file and st.button("Run Core Analysis"):
        holidays_df = pd.read_excel(holidays_file)
        data_df = pd.read_excel(data_file)
        data_df = _filter_by_date_range(data_df, date_start, date_end)
        # Call with signature-aware kwargs
        try:
            fig, df_out = call_with_supported_kwargs(
                core_deposit_analysis,
                data_df,
                holidays_df,
                branch=None,
                product=None,
                time_bucket=None,
                currency=currency,
                freq=freq,
                type=agg_type
            )
            st.pyplot(fig, use_container_width=True)
            st.dataframe(df_out)
        except Exception as e:
            st.exception(e)

elif analysis == "Prepayment Analysis":
    st.subheader("Inputs")
    start = st.text_input("start (Business_Days start, ISO)", value="2017-01-01")
    end = st.text_input("end (Business_Days end, ISO)", value="2025-12-31")
    currency = st.selectbox("currency", ["EUR", "USD", "TRY"], index=2, key="curr_prepay")

    holidays_file = st.file_uploader("Upload Turkey_Holidays.xlsx", type=["xlsx"], key="hol_prepay")
    data_file = st.file_uploader("Upload PREPAYMENT_DATA.xlsx", type=["xlsx"], key="data_prepay")

    if holidays_file and data_file and st.button("Run Prepayment Analysis"):
        holidays_df = pd.read_excel(holidays_file)
        data_df = pd.read_excel(data_file)
        data_df = _filter_by_date_range(data_df, start, end)
        business_days = pd.date_range(start=start, end=end, freq='B')

        try:
            report = call_with_supported_kwargs(
                data_model,
                data_df,
                data_year_basis=8,
                start_date=start,
                end_date=end,
                excel=False,
                business_days=business_days,
                holidays=holidays_df
            )
        except Exception as e:
            st.exception(e)
            report = None

        try:
            df = call_with_supported_kwargs(
                df_prepayment,
                report if report is not None else data_df,
                branch=None,
                product=None,
                time_bucket=None,
                currency=currency
            )

            table_a = None
            fig = None
            if isinstance(df, tuple) and len(df) == 2:
                table_a, fig = df
            else:
                table_a = df
                try:
                    fig, ax = plt.subplots()
                    pd.DataFrame(table_a).select_dtypes(include=[float, int]).plot(ax=ax)
                except Exception:
                    fig = None

            st.markdown("**Table A**")
            st.dataframe(pd.DataFrame(table_a) if not isinstance(table_a, pd.DataFrame) else table_a)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.exception(e)

else:  # Early Withdrawal Analysis
    st.subheader("Inputs")
    start_date = st.text_input("start_date (ISO)", value="2017-01-01", key="ew_start")
    end_date = st.text_input("end_date (ISO)", value="2025-12-31", key="ew_end")
    currency = st.selectbox("currency", ["EUR", "USD", "TRY"], index=2, key="curr_ew")
    product = st.text_input("product (e.g., 'Vadeli.Mevduat.Ticari')", value="Vadeli.Mevduat.Ticari")

    holidays_file = st.file_uploader("Upload Turkey_Holidays.xlsx", type=["xlsx"], key="hol_ew")
    data_file = st.file_uploader("Upload EARLY_WITHDRAWAL_DATA.xlsx", type=["xlsx"], key="data_ew")

    if holidays_file and data_file and st.button("Run Early Withdrawal Analysis"):
        holidays_df = pd.read_excel(holidays_file)
        data_df = pd.read_excel(data_file)
        data_df = _filter_by_date_range(data_df, start_date, end_date)
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')

        # Try to build a report object via data_model if present; otherwise fall back to simple dict
        try:
            report = call_with_supported_kwargs(
                data_model,
                data_df,
                data_year_basis=8,
                start_date=start_date,
                end_date=end_date,
                excel=False,
                business_days=business_days,
                holidays=holidays_df
            )
        except Exception:
            report = {"data": data_df, "holidays": holidays_df, "business_days": business_days}

        try:
            df = call_with_supported_kwargs(
                df_early_withdrawal,
                report,
                branch=None,
                product=product,
                time_bucket=None,
                currency=currency
            )

            table_a = None
            fig = None
            if isinstance(df, tuple) and len(df) == 2:
                table_a, fig = df
            else:
                table_a = df
                try:
                    fig, ax = plt.subplots()
                    pd.DataFrame(table_a).select_dtypes(include=[float, int]).plot(ax=ax)
                except Exception:
                    fig = None

            st.markdown("**Table A**")
            st.dataframe(pd.DataFrame(table_a) if not isinstance(table_a, pd.DataFrame) else table_a)
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.exception(e)

