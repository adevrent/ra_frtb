import math
import pyreadr
import warnings
import lifelines
import panel as pn
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pn.extension('echarts')
warnings.simplefilter('ignore')

wd = 'D:/AND.R.Analysis.Platform/AND.Risk.Management/Data.History/AND.SAS.Database.Daily/Raw.Data/'
wd_SAVE = r"C:\Users\adevr\riskactive_main\FRTB\From_Sait\Core Analysis"
wd_PRAM = r"C:\Users\adevr\riskactive_main\FRTB\From_Sait"

progress_bar = Progress(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn(),
                        TextColumn("•"),
                        TimeRemainingColumn())

core_deposit_filepath = r"C:\Users\adevr\riskactive_main\FRTB\From_Sait\Core Analysis\CORE_DEPOSIT_DATA.xlsx"
df = pd.DataFrame()

with progress_bar as pb:
    df = pd.read_excel(core_deposit_filepath)

df.columns = df.columns.astype(str).str.replace(".", "_")
# print(df)

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

last_date_of_report = pd.to_datetime(df["REPORT_DATE"]).max()
last_date_of_month = last_date_of_report.replace(month=last_date_of_report.month+1, day=1) - timedelta(days=1)
Monthly_Business_Days = Monthly_Business_Days[(Monthly_Business_Days['last'] <= last_date_of_month)]

Rolling_Dates = rolling_dates(Monthly_Business_Days, year = 8)

# ['Vadeli.Mevduat.Tasarruf', 'Vadeli.Mevduat.Ticari']
# ['TRY', 'CHF', 'EUR', 'USD', 'JPY', 'GBP', 'SEK', 'CAD', 'ZAR', 'DKK', 'AUD', 'NOK', 'RUB', 'MXN', 'XAU', 'QAR', 'XAG']

Report_Core_Deposit = data_model(df, 
                                     data_year_basis = 8,
                                     start_date = None,
                                     end_date = None,
                                     excel = False)


Report_Core_Deposit.drop(Report_Core_Deposit[Report_Core_Deposit["REPORT_DATE"] == "2020-12-08"].index, inplace=True)
Report_Core_Deposit.drop(Report_Core_Deposit[Report_Core_Deposit["REPORT_DATE"] == "2023-12-27"].index, inplace=True)

Report_Core_Deposit.rename(columns={'REPORT_DATE': 'Report_Date', 'BRANCH': 'Branch', 'CUSTOMER_NO': 'Customer_No', 'CURRENCY': 'Currency',
                                        'PRODUCT_CODE': 'Product_Code', 'NOTIONAL': 'Notional'}, inplace=True)     
Report_Core_Deposit['Time_Bucket'] = 0
Report_Core_Deposit = (Report_Core_Deposit.groupby(['Report_Date','Branch', 'Customer_No', 'Currency', 'Product_Code', 'Time_Bucket'], as_index=False)
                           .agg({"Notional": "sum"})
                           .reset_index(drop=True))
Report_Core_Deposit = Report_Core_Deposit[Report_Core_Deposit['Notional']>0]
#Report_Core_Deposit.to_excel(wd_SAVE+'Report_Core_Deposit_BS.xlsx', index=False)

Report_Core_Deposit[Report_Core_Deposit.isnull().any(axis=1)].sort_values(by='Report_Date')

tmp = Report_Core_Deposit.copy()

# Normalize for inspection (no drops yet)
for c in ['Product_Code','Currency','Branch','Time_Bucket']:
    if c in tmp.columns and tmp[c].dtype == 'object':
        tmp[c] = (tmp[c].astype(str)
                          .str.replace('\u00A0',' ', regex=False)  # NBSP → space
                          .str.normalize('NFKC')                  # fix unicode width
                          .str.strip())

target = 'Vadesiz.Mevduat.Tasarruf'

# Now combine with Currency
m1 = tmp['Currency'].eq('TRY')
m2 = tmp['Product_Code'].eq(target)
m2c = tmp['Product_Code'].str.contains(target, regex=False, na=False)

# Date parsing check
tmp['Report_Date'] = pd.to_datetime(tmp['Report_Date'], errors='coerce')

core_deposit_analysis(Report_Core_Deposit, 
                      branch = None,
                      product = "Vadesiz.Mevduat.Ticari",
                      time_bucket = None,
                      currency = 'TRY',
                      freq = 'W', type = 'mean', nsim = 100000, excel = False, plot = True)