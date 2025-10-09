
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="FRTB Analyses", layout="wide")

st.title("FRTB Analyses — Core / Prepayment / Early Withdrawal")
st.caption("Upload-only app: no static filepaths. Drop your Turkey_Holidays.xlsx and the corresponding data file as required by each analysis.")

# Notebook-derived imports
import math
import pyreadr
import warnings
import lifelines
import panel as pn
import numpy as np
import pandas as pd
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


# ---- extracted from notebooks ----
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

# ---- extracted from notebooks ----
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

# ---- extracted from notebooks ----
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

# ---- extracted from notebooks ----
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
        try:
            holidays_df = pd.read_excel(holidays_file)
            data_df = pd.read_excel(data_file)
            # Call the notebook function
            fig, df_out = core_deposit_analysis(
                data_df,
                holidays_df,
                date_start=date_start,
                date_end=date_end,
                branch=None,
                product=None,
                time_bucket=None,
                currency=currency,
                freq=freq,
                type=agg_type
            )
            st.pyplot(fig, use_container_width=True)
            st.dataframe(df_out)
        except NotImplementedError as e:
            st.error(str(e))
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
        try:
            holidays_df = pd.read_excel(holidays_file)
            data_df = pd.read_excel(data_file)

            # Build Business_Days and feed to data model per user instruction
            business_days = pd.date_range(start=start, end=end, freq='B')

            report = data_model(
                data_df,
                data_year_basis=8,
                start_date=start,
                end_date=end,
                excel=False
            )

            # df_prepayment call; branch/product/time_bucket are None; currency is user-selected
            df = df_prepayment(
                report,
                branch=None,
                product=None,
                time_bucket=None,
                currency=currency
            )

            # Expecting table_a and a figure from the notebook logic. If df_prepayment returns tuple, unpack.
            # Try to be flexible:
            table_a = None
            fig = None
            if isinstance(df, tuple) and len(df) == 2:
                table_a, fig = df
            else:
                # If single object returned, assume its the table and build a basic plot if possible
                table_a = df
                try:
                    fig, ax = plt.subplots()
                    table_a.select_dtypes(include=[float, int]).plot(ax=ax)
                except Exception:
                    fig = None

            st.markdown("**Table A**")
            if isinstance(table_a, pd.DataFrame):
                st.dataframe(table_a)
            else:
                try:
                    st.dataframe(pd.DataFrame(table_a))
                except Exception:
                    st.write(table_a)

            if fig is not None:
                st.pyplot(fig, use_container_width=True)
        except NotImplementedError as e:
            st.error(str(e))
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
        try:
            holidays_df = pd.read_excel(holidays_file)
            data_df = pd.read_excel(data_file)

            business_days = pd.date_range(start=start_date, end=end_date, freq='B')

            # Reconstruct an object similar to notebook's Report_Early_Withdrawal if required.
            # Many notebooks create such a report via a modeling function; if present, use it.
            # If df_early_withdrawal expects a prepared 'report' structure, try to build it via data_model if it generalizes.
            try:
                report = data_model(
                    data_df,
                    data_year_basis=8,
                    start_date=start_date,
                    end_date=end_date,
                    excel=False
                )
            except Exception:
                report = {"data": data_df, "holidays": holidays_df, "business_days": business_days}

            df = df_early_withdrawal(
                report,
                branch=None,
                product=product,
                time_bucket=None,
                currency=currency
            )

            # Similar flexibility as above
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
            if isinstance(table_a, pd.DataFrame):
                st.dataframe(table_a)
            else:
                try:
                    st.dataframe(pd.DataFrame(table_a))
                except Exception:
                    st.write(table_a)

            if fig is not None:
                st.pyplot(fig, use_container_width=True)
        except NotImplementedError as e:
            st.error(str(e))
        except Exception as e:
            st.exception(e)

