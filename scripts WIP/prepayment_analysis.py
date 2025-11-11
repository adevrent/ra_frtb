import numpy as np
import pandas as pd
import pyreadr
import warnings
import lifelines
import matplotlib.pyplot as plt

from scipy import stats
from datetime import timedelta
from timeit import default_timer as timer
from itertools import combinations, batched
from pathlib import Path
from tqdm import tqdm
from rich.progress import track
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sksurv.nonparametric import kaplan_meier_estimator
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from lifelines import KaplanMeierFitter, WeibullAFTFitter, CoxPHFitter
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter, PiecewiseExponentialFitter, NelsonAalenFitter, SplineFitter
from lifelines.utils import find_best_parametric_model, median_survival_times

from IPython.display import display, HTML, Markdown
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.simplefilter('ignore')

# Dosyaları aslında user yükleyecek.
wd = r"C:\Users\adevr\ra_frtb\data"
wd_SAVE = r"C:\Users\adevr\ra_frtb\data"
wd_PRAM = r"C:\Users\adevr\ra_frtb\data"

prepayment_filepath = r"C:\Users\adevr\ra_frtb\data\PREPAYMENT_DATA.xlsx"
data = pd.read_excel(prepayment_filepath)
data.columns = data.columns.astype(str).str.replace(".", "_")

Turkey_Holidays = pd.read_excel(f"{wd_PRAM}/Turkey_Holidays.xlsx")
Turkey_Holidays['TURKEY_HOLIDAYS'] = pd.to_datetime(Turkey_Holidays['TURKEY_HOLIDAYS'])

Business_Days = pd.date_range(start="2015-01-01", end="2035-12-31", freq='B')
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

Max_Report_Date = (
    data.groupby('CUSTOMER_ID')['REPORT_DATE']
    .agg(MAX_RPT='max', 
         N_RPT='size')
    .reset_index()
)

Max_Report_Date['CUSTOMER_IDs'] = Max_Report_Date.loc[:, 'CUSTOMER_ID']
Max_Report_Date.set_index('CUSTOMER_IDs', inplace=True)
Max_Report_Date.index.names = ['CUSTOMER_ID']


last_date_of_report = max(Max_Report_Date['MAX_RPT'])

last_date_of_month = last_date_of_report.replace(month=last_date_of_report.month+1, day=1) - timedelta(days=1)

Monthly_Business_Days = Monthly_Business_Days[(Monthly_Business_Days['last'] <= last_date_of_month)]

Rolling_Dates = rolling_dates(Monthly_Business_Days, year = 8)

Report_Prepayment = data_model(data, 
                                   data_year_basis = 8, 
                                   start_date = None,                         
                                   end_date = None,                           
                                   excel = False)

Prepayment_Analysis = ( 
    Report_Prepayment.groupby(['Branch','Currency','Product_Code','Time_Bucket'])["Prepayment_Notional"].sum()
    .div(Report_Prepayment.groupby(['Branch','Currency','Product_Code','Time_Bucket'])['Notional'].sum())
    .reset_index(name="Prepayment_Ratio")
    .round(4)
)

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

df = df_prepayment(Report_Prepayment, 
                   branch = None, 
                   product = None, 
                   time_bucket = None , 
                   currency = 'TRY')

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

fig.show()

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
display(HTML(table_a.to_html(index=False)))

print(table_a)