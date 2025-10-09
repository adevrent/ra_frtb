import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sksurv.nonparametric import kaplan_meier_estimator
from IPython.display import HTML, display

def run_km_prepayment(df):
    time_points, survival_probabilities, confidence_interval, median_survival_time = kaplan_meier_estimator_custom(df)

    survival_probabilities_diff = survival_probabilities - 0.5
    median_survival_time_adj = np.interp(min(survival_probabilities_diff), survival_probabilities_diff[::-1], time_points[::-1])

    if max(survival_probabilities) > 0.5:  
        median_survival_time = median_survival_time
    else:
        median_survival_time = median_survival_time_adj

    fig = make_subplots(rows=1, cols=1, subplot_titles=(""))

    fig.add_trace(go.Scatter(
        x=time_points,
        y=survival_probabilities,
        mode='lines',
        line=dict(color='darkorange'),
        name='Notional / Core Deposit'),
                  row=1, col=1
                 )

    y_min = round(min(survival_probabilities),2)

    fig.update_layout(
                title="Prepayment Analysis",
                width=700,
                height=500,
                plot_bgcolor="#fff",
                showlegend=False,
                yaxis=dict(range=[y_min,1])
    ) 
        

    fig.add_vline(x=median_survival_time_adj, line_width=2, line_dash="dot", line_color="royalblue")

    fig.add_vline(x=365.25, line_width=2, line_dash="dot", line_color="firebrick")
    fig.add_hline(y=round((np.interp(365.25, time_points[::1], survival_probabilities[::1])),2), line_width=2, line_dash="dot", line_color="firebrick")


    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Probability of Survival", row=1, col=1)


    table_a = {'1-year Survival Probability (95% CI)': [],
               '1-year Prepayment Probability (95% CI)': [],
               'Survival Time (Adjusted)': [],
               'Survival Time (Probability)': [],
               'Survival Time (Max)': []}

    table_a['1-year Survival Probability (95% CI)'].append(round((np.interp(365.25, time_points[::1], survival_probabilities[::1]))*100,2))
    table_a['1-year Prepayment Probability (95% CI)'].append(round((1-np.interp(365.25, time_points[::1], survival_probabilities[::1]))*100,2))
    table_a['Survival Time (Adjusted)'].append(round(median_survival_time_adj,0))
    table_a['Survival Time (Probability)'].append(round((np.interp(median_survival_time_adj, time_points[::1], survival_probabilities[::1]))*100,2))
    table_a['Survival Time (Max)'].append(round(median_survival_time,0))

    table_a = pd.DataFrame(table_a)
    return fig, table_a

def run_km_early_withdrawal(df):
    time_points, survival_probabilities, confidence_interval, median_survival_time = kaplan_meier_estimator_custom(df)

    survival_probabilities_diff = survival_probabilities - 0.5
    median_survival_time_adj = np.interp(min(survival_probabilities_diff), survival_probabilities_diff[::-1], time_points[::-1])

    if max(survival_probabilities) > 0.5:  
        median_survival_time = median_survival_time
    else:
        median_survival_time = median_survival_time_adj

    fig = make_subplots(rows=1, cols=1, subplot_titles=(""))

    fig.add_trace(go.Scatter(
        x=time_points,
        y=survival_probabilities,
        mode='lines',
        line=dict(color='darkorange'),
        name='Notional / Core Deposit'),
                  row=1, col=1
                 )

    y_min = round(min(survival_probabilities),2)

    fig.update_layout(
                title="Early Withdrawal Analysis",
                width=700,
                height=500,
                plot_bgcolor="#fff",
                showlegend=False,
                yaxis=dict(range=[y_min,1])
    ) 


    fig.add_vline(x=median_survival_time_adj, line_width=2, line_dash="dot", line_color="royalblue")

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Probability of Survival", row=1, col=1)


    table_a = {'1-year Survival Probability (95% CI)': [],
               '1-year Early Withdrawal Probability (95% CI)': [],
               'Survival Time (Adjusted)': [],
               'Survival Time (Probability)': [],
               'Survival Time (Max)': []}

    table_a['1-year Survival Probability (95% CI)'].append(round((np.interp(365.25, time_points[::1], survival_probabilities[::1]))*100,2))
    table_a['1-year Early Withdrawal Probability (95% CI)'].append(round((1-np.interp(365.25, time_points[::1], survival_probabilities[::1]))*100,2))
    table_a['Survival Time (Adjusted)'].append(round(median_survival_time_adj,0))
    table_a['Survival Time (Probability)'].append(round((np.interp(median_survival_time_adj, time_points[::1], survival_probabilities[::1]))*100,2))
    table_a['Survival Time (Max)'].append(round(median_survival_time,0))

    table_a = pd.DataFrame(table_a)
    return fig, table_a

# === Core Analysis functions (patched) ===

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

def data_model_core(data, data_year_basis = None, start_date = None, end_date = None, excel = False):
    
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
        
        # fig.show()

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
    
    # display(HTML(table_a.to_html(index=False)))
    # display(HTML(table_b.to_html(index=False)))
    # display(HTML(table_c.to_html(index=False)))
    return table_a, table_b, table_c, data, (fig if 'fig' in locals() else None)
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

# === Prepayment helpers ===


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



# === Prepayment data_model (renamed) ===

def data_model_prep(data, data_year_basis = None, start_date = None, end_date = None, excel = False, Max_Report_Date = None):
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

# === df_prepayment ===

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

# === Early Withdrawal helpers ===

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
    


# === Early Withdrawal data_model (renamed) ===

def data_model_ew(data, data_year_basis = None, start_date = None, end_date = None, excel = False):
    
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

# === df_early_withdrawal ===

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

def build_business_days(holidays_df, start, end):
    holidays_series = pd.to_datetime(holidays_df.iloc[:,0])
    business_days = pd.date_range(start=start, end=end, freq='B')
    turkey_business_days = business_days[~business_days.isin(holidays_series)]
    monthly_bd = (
        pd.DataFrame({"Business_Days": turkey_business_days})
        .assign(YearMonth=lambda df: df["Business_Days"].dt.to_period("M"))
        .groupby("YearMonth")["Business_Days"]
        .agg(["first", "last"])
        .reset_index(drop=True)
    )
    return turkey_business_days, monthly_bd


st.set_page_config(page_title="FRTB Analyses App", layout="wide")

st.title("FRTB Analyses — Core Deposit / Prepayment / Early Withdrawal")

analysis = st.radio("Choose analysis", ["Core Analysis", "Prepayment Analysis", "Early Withdrawal Analysis"], horizontal=True)

holidays_file = st.file_uploader("Upload Turkey Holidays XLSX", type=["xlsx"], key="holidays")

if analysis == "Core Analysis":
    data_file = st.file_uploader("Upload CORE_DEPOSIT_DATA.xlsx", type=["xlsx"], key="core")
    col1, col2, col3 = st.columns(3)
    with col1:
        date_start = st.text_input("Start date (ISO)", "2017-01-01")
    with col2:
        date_end = st.text_input("End date (ISO)", "2018-01-01")
    with col3:
        currency = st.selectbox("Currency", ["USD", "EUR", "TRY"], index=2)
    col4, col5 = st.columns(2)
    with col4:
        freq = st.selectbox("Frequency", ["W", "M", "Y"], index=1)
    with col5:
        agg_type = st.selectbox("Aggregation type", ["mean", "roll"], index=0)
    run = st.button("Run Core Analysis", type="primary", use_container_width=True)

    if run:
        if holidays_file is None or data_file is None:
            st.error("Please upload both the Turkey Holidays file and the CORE_DEPOSIT_DATA.xlsx file.")
        else:
            holidays_df = pd.read_excel(holidays_file)
            data_df = pd.read_excel(data_file)
            # Ensure dates are datetime
            try:
                data_df['REPORT_DATE'] = pd.to_datetime(data_df['REPORT_DATE'])
            except Exception:
                pass

            try:
                # core_deposit_analysis returns (table_a, table_b, table_c, data, fig_or_none) after our patch
                table_a, table_b, table_c, out_data, fig = core_deposit_analysis(
                    data=data_df,
                    branch=None,
                    product=None,
                    time_bucket=None,
                    currency=currency,
                    freq=freq,
                    type=agg_type,
                    nsim=100000,
                    excel=False,
                    plot=True
                )
                st.subheader("Core Deposit Plot")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Figure not available from function.")
                st.subheader("Tables")
                st.dataframe(table_a, use_container_width=True)
                st.dataframe(table_b, use_container_width=True)
                st.dataframe(table_c, use_container_width=True)
            except Exception as e:
                st.exception(e)

elif analysis == "Prepayment Analysis":
    data_file = st.file_uploader("Upload PREPAYMENT_DATA.xlsx", type=["xlsx"], key="prep")
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.text_input("Business Days start (ISO)", "2017-01-01")
    with col2:
        end = st.text_input("Business Days end (ISO)", "2025-12-31")
    with col3:
        currency = st.selectbox("Currency", ["EUR", "USD", "TRY"], index=2)
    run = st.button("Run Prepayment Analysis", type="primary", use_container_width=True)
    if run:
        if holidays_file is None or data_file is None:
            st.error("Please upload both the Turkey Holidays file and the PREPAYMENT_DATA.xlsx file.")
        else:
            holidays_df = pd.read_excel(holidays_file)
            data_df = pd.read_excel(data_file)
            # Build business days frames
            turkey_bd, monthly_bd = build_business_days(holidays_df, start, end)
            # The data_model_prep expects 'Monthly_Business_Days' to be available; we will set it as a global
            globals()['Monthly_Business_Days'] = monthly_bd

            try:
                Report_Prepayment = data_model_prep(
                    data_df,
                    data_year_basis=8,
                    start_date=pd.to_datetime(start),
                    end_date=pd.to_datetime(end),
                    excel=False
                )
                df = df_prepayment(Report_Prepayment, branch=None, product=None, time_bucket=None, currency=currency)
                # Run the notebook's analysis cell logic
                # We rely on kaplan_meier_estimator_custom and plotly imports already included
                # Variables 'df', 'np', 'make_subplots', 'go' are available
                # Execute prepayment analysis code in a local namespace
                fig, table_a = run_km_prepayment(df)
                st.subheader("Prepayment Survival Plot")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No figure produced.")
                st.subheader("Table A")
                if isinstance(table_a, dict):
                    table_a = pd.DataFrame(table_a)
                st.dataframe(table_a, use_container_width=True)
            except Exception as e:
                st.exception(e)

else:  # Early Withdrawal Analysis
    data_file = st.file_uploader("Upload EARLY_WITHDRAWAL_DATA.xlsx", type=["xlsx"], key="ew")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.text_input("Start date (ISO)", "2017-01-01")
    with col2:
        end_date = st.text_input("End date (ISO)", "2025-12-31")
    with col3:
        currency = st.selectbox("Currency", ["EUR", "USD", "TRY"], index=2)
    run = st.button("Run Early Withdrawal Analysis", type="primary", use_container_width=True)
    if run:
        if holidays_file is None or data_file is None:
            st.error("Please upload both the Turkey Holidays file and the EARLY_WITHDRAWAL_DATA.xlsx file.")
        else:
            holidays_df = pd.read_excel(holidays_file)
            data_df = pd.read_excel(data_file)
            turkey_bd, monthly_bd = build_business_days(holidays_df, start_date, end_date)
            globals()['Monthly_Business_Days'] = monthly_bd
            try:
                Report_Early_Withdrawal = data_model_ew(
                    data_df,
                    data_year_basis=None,
                    start_date=pd.to_datetime(start_date),
                    end_date=pd.to_datetime(end_date),
                    excel=False
                )
                df = df_early_withdrawal(Report_Early_Withdrawal, branch=None, product='Vadeli.Mevduat.Ticari', time_bucket=None, currency=currency)
                # Execute EW analysis cell logic
                fig, table_a = run_km_early_withdrawal(df)
                st.subheader("Early Withdrawal Survival Plot")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No figure produced.")
                st.subheader("Table A")
                if isinstance(table_a, dict):
                    table_a = pd.DataFrame(table_a)
                st.dataframe(table_a, use_container_width=True)
            except Exception as e:
                st.exception(e)