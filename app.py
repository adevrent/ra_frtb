import io
import sys
import math
import warnings
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

import streamlit as st

# Optional/extra deps used by original scripts — if missing, we still proceed where possible
try:
    import statsmodels.api as sm
except Exception as e:
    sm = None

try:
    from sksurv.nonparametric import kaplan_meier_estimator
except Exception as e:
    kaplan_meier_estimator = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    go = None
    make_subplots = None

warnings.simplefilter('ignore')

# --------------------------- Shared helpers ---------------------------

def _parse_date(s):
    if s is None or str(s).strip() == "":
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None

def _none_if_empty(s):
    if s is None:
        return None
    s = str(s).strip()
    return s if s != "" else None

def build_business_cal(start_date, end_date, turkey_holidays_df):
    """
    Build Turkey business day calendar & monthly first/last table.
    turkey_holidays_df must contain a column named 'TURKEY_HOLIDAYS' in datetime-like format.
    """
    th = turkey_holidays_df.copy()
    if 'TURKEY_HOLIDAYS' not in th.columns:
        # Try to accept first column as holidays if named differently
        th.columns = [str(c).strip() for c in th.columns]
        if len(th.columns) > 0:
            th = th.rename(columns={th.columns[0]: 'TURKEY_HOLIDAYS'})
        if 'TURKEY_HOLIDAYS' not in th.columns:
            raise ValueError("Turkey_Holidays.xlsx must include a 'TURKEY_HOLIDAYS' column.")
    th['TURKEY_HOLIDAYS'] = pd.to_datetime(th['TURKEY_HOLIDAYS']).dt.date

    # Defaults if None
    if start_date is None:
        start_date = datetime(2010, 1, 1).date()
    if end_date is None:
        end_date = datetime(2030, 12, 31).date()

    business_days = pd.bdate_range(start=start_date, end=end_date).date
    # Exclude holidays
    business_days = pd.to_datetime([d for d in business_days if d not in set(th['TURKEY_HOLIDAYS'])])

    monthly_bd = (
        pd.DataFrame({"Business_Days": business_days})
        .assign(YearMonth=lambda df: df["Business_Days"].dt.to_period("M"))
        .groupby("YearMonth")["Business_Days"]
        .agg(["first", "last"])
        .reset_index()
    )
    return th, monthly_bd

def next_business_day(date_val, holidays_df):
    """Next business day after date_val given holidays_df with first column holidays."""
    if isinstance(holidays_df, pd.DataFrame):
        if holidays_df.shape[1] == 0:
            holidays = set()
        else:
            holidays = set(pd.to_datetime(holidays_df.iloc[:,0]).dt.date)
    else:
        holidays = set()

    if pd.isna(date_val):
        return date_val
    date_val = pd.to_datetime(date_val).date()
    next_day = date_val + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in holidays:
        next_day += timedelta(days=1)
    return next_day

def classify_time_bucket_core(months):
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

def classify_time_bucket_prepayment(months):
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

def rolling_dates(monthly_bd_df, years=5):
    period = years * 12
    rows = []
    for i in range(len(monthly_bd_df) - (period - 1)):
        start_d = monthly_bd_df.loc[i, "first"]
        end_d = monthly_bd_df.loc[i + (period - 1), "last"]
        rows.append({"START_DATE": start_d, "END_DATE": end_d})
    return pd.DataFrame(rows)

def kaplan_meier_estimator_custom(df):
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    if kaplan_meier_estimator is None:
        raise ImportError("sksurv is required for Kaplan-Meier estimation. Please install 'scikit-survival'.")

    # Note: original scripts use (event, time) in reversed order; sksurv expects (event, time)
    kme = kaplan_meier_estimator(df["Status_TF"], df["Survival_in_Days"], conf_type="log-log")
    time_points = kme[0]
    survival_probabilities = kme[1]
    confidence_interval = kme[2]
    median_survival_time = np.interp(0.5, survival_probabilities[::-1], time_points[::-1])
    survival_probabilities_diff = survival_probabilities - 0.5
    median_survival_time_adj = np.interp(min(survival_probabilities_diff), survival_probabilities_diff[::-1], time_points[::-1])
    if min(survival_probabilities) > 0.5:
        median_survival_time = median_survival_time_adj
    return time_points, survival_probabilities, confidence_interval, median_survival_time, median_survival_time_adj

# --------------------------- Core Deposit analysis (from core_analysis.py) ---------------------------

def data_model_core(df, monthly_bd_df, data_year_basis=None, start_date=None, end_date=None):
    df = df.copy()
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"])

    if data_year_basis is not None:
        df_rd = rolling_dates(monthly_bd_df, years=data_year_basis)
        start_date = df_rd["START_DATE"].iloc[-1]
        end_date = df_rd["END_DATE"].iloc[-1]

    if start_date is not None and end_date is not None:
        df = df[(df["REPORT_DATE"] >= pd.to_datetime(start_date)) & (df["REPORT_DATE"] <= pd.to_datetime(end_date))].reset_index(drop=True)

    return df

def reshape_core(df):
    df = df.rename(columns={
        'REPORT_DATE': 'Report_Date',
        'BRANCH': 'Branch',
        'CUSTOMER_NO': 'Customer_No',
        'CURRENCY': 'Currency',
        'PRODUCT_CODE': 'Product_Code',
        'NOTIONAL': 'Notional'
    })
    df["Time_Bucket"] = 0
    df = (df.groupby(['Report_Date','Branch','Customer_No','Currency','Product_Code','Time_Bucket'], as_index=False)
            .agg({"Notional": "sum"})
            .reset_index(drop=True))
    df = df[df['Notional'] > 0]
    return df

def filter_core(df, branch=None, product=None, time_bucket=None, currency=None):
    data = df.copy()
    # Apply the same filter logic as original
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == int(time_bucket)) & (data['Currency'] == currency)]
    elif branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == int(time_bucket)) & (data['Currency'] == currency)]
    elif branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == int(time_bucket)) & (data['Currency'] == currency)]
    elif branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
    elif branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == int(time_bucket)) & (data['Currency'] == currency)]
    elif branch is None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Currency'] == currency)]
    elif branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
    elif branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
    data = (data.groupby(['Report_Date'], as_index=False).agg({"Notional": "sum"}).reset_index(drop=True))
    return data

def core_deposit_analysis(df_series, freq='W', type_='last', nsim=100000):
    """Replicates core_deposit_analysis logic that builds trend/core and persistence."""
    if df_series.empty:
        raise ValueError("The filtered data is empty. Please check the filter criteria.")
    if sm is None:
        raise ImportError("statsmodels is required for core analysis (HP filter). Please install 'statsmodels'.")
    data = df_series.copy().set_index("Report_Date")

    # Resampling
    if freq == 'W':
        if type_ == 'mean':
            data = data.resample('W').mean()
        elif type_ == 'roll':
            data['Notional'] = data['Notional'].rolling(window=5).mean()
            data = data.dropna()
        else:
            data = data.resample('W').last()
    elif freq == 'M':
        if type_ == 'mean':
            data = data.resample('M').mean()
        elif type_ == 'roll':
            data['Notional'] = data['Notional'].rolling(window=21).mean()
            data = data.dropna()
        else:
            data = data.resample('M').last()
    elif freq == 'Y':
        if type_ == 'mean':
            data = data.resample('Y').mean()
        else:
            data = data.resample('Y').last()

    period = pd.to_datetime(data.index)
    arr = data['Notional'].values.reshape(-1, 1)
    df_log = np.log(arr)

    # HP filter lambda choices from original
    if freq == 'Y':
        hp_lambda = 100
    elif freq == 'M':
        hp_lambda = 14400
    elif freq == 'W':
        if type_ == 'mean':
            hp_lambda = 125000
        elif type_ == 'roll':
            hp_lambda = 129600
        else:
            hp_lambda = 125000
    elif freq == 'D':
        hp_lambda = 129600
    else:
        hp_lambda = 129600

    cycle, trend = sm.tsa.filters.hpfilter(df_log, lamb=hp_lambda)

    df_sigma = np.std(df_log[:, 0] - trend)
    from scipy.stats import norm  # local import to avoid global missing
    df_log_core = trend - norm.ppf(0.99, loc=0, scale=1) * df_sigma

    df_core = np.exp(df_log_core)
    df_core_percentage = df_core[-1] / np.exp(trend[-1])
    df_sigma = df_sigma * (1 - df_core_percentage)

    # Monte Carlo-like path (deterministic formula used in original)
    df_log_MPA = np.zeros((nsim + 1, 1))
    df_log_MPA[0] = df_log_core[-1]
    for j in range(1, nsim + 1):
        df_log_MPA[j] = df_log_core[-1] - 0.5 * df_sigma**2 * j - df_sigma * np.sqrt(j) * norm.ppf(0.99)
    df_MPA = np.exp(df_log_MPA)

    df_MPCF = np.zeros((nsim,))
    for j in range(nsim):
        df_MPCF[j] = df_MPA[j] - df_MPA[j + 1]

    width = (df_MPA[:-1] + df_MPA[1:]) / 2
    mean_life = np.sum(width) / df_MPA[0]
    horizon = int(round(mean_life.item()))

    df_MPCF_adj = np.zeros((horizon,))
    for j in range(horizon):
        df_MPCF_adj[j] = df_MPCF[j] + df_MPA[horizon] / horizon

    df_MPA_adj = np.zeros((horizon + 1,))
    df_MPA_adj[0] = df_MPA[0]
    for j in range(horizon):
        df_MPA_adj[j + 1] = df_MPA_adj[j] - df_MPCF_adj[j]

    persistence = df_MPA_adj / df_MPA_adj[0]

    persistence2 = np.ones(horizon + 1)
    for j in range(1, horizon + 1):
        persistence2[j] = persistence[j - 1] * df_core_percentage

    distribution = persistence2[:-1] - persistence2[1:]
    average_life = float(np.dot(distribution, np.arange(len(distribution))))
    horizon_life = horizon
    core_amount = float(df_core[-1])

    # Reporting by frequency mapping (to days/months/years) — same as originals
    def freq_to_days_months_years(avg_life, horiz):
        digit = 2
        if freq == 'D':
            return round(avg_life, 0), round(avg_life/21, digit), round(avg_life/252, digit), round(horiz, 0), round(horiz/21, digit), round(horiz/252, digit)
        if freq == 'W':
            if type_ == 'roll':
                # 'roll' uses raw bar count as days (1 bar ~1 day) in original;
                # however original maps 'roll' to same as 'mean' for W->days
                mean_days = round(avg_life, 0)
                mean_months = round(avg_life/21, digit)
                mean_years = round(avg_life/252, digit)
                horizon_days = round(horiz, 0)
                horizon_months = round(horiz/21, digit)
                horizon_years = round(horiz/252, digit)
            else:
                mean_days = round(avg_life*5/1, 0)
                mean_months = round(avg_life*5/21, digit)
                mean_years = round(avg_life*5/252, digit)
                horizon_days = round(horiz*5/1, 0)
                horizon_months = round(horiz*5/21, digit)
                horizon_years = round(horiz*5/252, digit)
            return mean_days, mean_months, mean_years, horizon_days, horizon_months, horizon_years
        if freq == 'M':
            return round(avg_life*21, 0), round(avg_life, digit), round(avg_life/12, digit), round(horiz*21, 0), round(horiz, digit), round(horiz/12, digit)
        if freq == 'Y':
            return round(avg_life/252, 0), round(avg_life/12, digit), round(avg_life, digit), round(horiz/252, 0), round(horiz/12, digit), round(horiz, digit)
        # default daily-like
        return round(avg_life, 0), round(avg_life/21, digit), round(avg_life/252, digit), round(horiz, 0), round(horiz/21, digit), round(horiz/252, digit)

    md, mm, my, hd, hm, hy = freq_to_days_months_years(average_life, horizon_life)

    # For plotting
    min_month_year = data.index.min().strftime('%b-%Y')
    max_month_year = data.index.max().strftime('%b-%Y')

    return {
        "df_resampled": data,
        "df_core": df_core,
        "core_pct": float(df_core_percentage.item() if hasattr(df_core_percentage, "item") else df_core_percentage),
        "core_amount": core_amount,
        "mean_days": md, "mean_months": mm, "mean_years": my,
        "horizon_days": hd, "horizon_months": hm, "horizon_years": hy,
        "min_month_year": min_month_year, "max_month_year": max_month_year,
        "persistence_days": int(hd if isinstance(hd, (int, np.integer)) else round(hd)),
        "persistence_series": persistence
    }

# --------------------------- Prepayment analysis (from prepayment_analysis.py) ---------------------------

def data_model_prepayment(data, monthly_bd_df, turkey_holidays_df, data_year_basis=None, start_date=None, end_date=None):
    # Max report date per customer
    Max_Report_Date = (
        data.groupby('CUSTOMER_ID')['REPORT_DATE']
        .agg(MAX_RPT='max', N_RPT='size')
        .reset_index()
    )
    Max_Report_Date['CUSTOMER_IDs'] = Max_Report_Date['CUSTOMER_ID']
    Max_Report_Date = Max_Report_Date.set_index('CUSTOMER_IDs')
    Max_Report_Date.index.names = ['CUSTOMER_ID']

    data = data.copy()
    data['REPORT_DATE'] = pd.to_datetime(data['REPORT_DATE'])

    if data_year_basis is not None:
        df_rolling_dates = rolling_dates(monthly_bd_df, years=data_year_basis)
        start_date = df_rolling_dates['START_DATE'].iloc[-1]
        end_date = df_rolling_dates['END_DATE'].iloc[-1]
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)].reset_index(drop=True)

    if start_date is not None and end_date is not None:
        data = data[(data['REPORT_DATE'] >= pd.to_datetime(start_date)) & (data['REPORT_DATE'] <= pd.to_datetime(end_date))].reset_index(drop=True)

    currency_series = data["CURRENCY"].copy()

    # Align to last report per customer then compute fields
    def _nx(d): return next_business_day(d, turkey_holidays_df)

    data = (
        data.assign(
            NO=lambda x: np.where(x['REPORT_DATE'] == x['CUSTOMER_ID'].map(Max_Report_Date.set_index('CUSTOMER_ID')['MAX_RPT']), 1, 0)
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
            MATURITY_CONT=lambda x: np.where(pd.to_datetime(x['MATURITY']).dt.weekday >= 5, 1, 0),
            MATURITY_NEW=lambda x: np.where(
                pd.to_datetime(x['MATURITY']).dt.weekday == 5, pd.to_datetime(x['MATURITY']) + pd.Timedelta(days=2),
                np.where(pd.to_datetime(x['MATURITY']).dt.weekday == 6, pd.to_datetime(x['MATURITY']) + pd.Timedelta(days=1), pd.to_datetime(x['MATURITY']))
            ),
            NEXT_BUS_DATE=lambda x: pd.to_datetime(x['REPORT_DATE']).apply(lambda d: _nx(d)),
            Maturity_Periods_D=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days,
            Maturity_Periods_M=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days / (365.25 / 12),
            SURV_TIME=lambda x: (pd.to_datetime(x['NEXT_BUS_DATE']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days,
            DIFF_TIME=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['NEXT_BUS_DATE'])).dt.days,
            DIFF_MONTH=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - (pd.to_datetime(x['MATURITY_NEW']) - pd.DateOffset(months=1))).dt.days,
            TIME_BUCKET=lambda x: x['Maturity_Periods_M'].apply(classify_time_bucket_prepayment)
        )
        .assign(
            Prepayment_Status=lambda x: np.where(
                (pd.to_datetime(x['NEXT_BUS_DATE']) < pd.to_datetime(x['MATURITY_NEW'])) & (x['DIFF_TIME'] > x['DIFF_MONTH']),
                "Early", "OnTime"
            )
        )
    )

    data["CURRENCY"] = currency_series.loc[data.index].values
    data = data[data['DIFF_TIME'] >= 0]

    data[['BRANCH', 'CUSTOMER_NO','CUSTOMER_EX_NO', 'PRODUCT_CODE']] = (data['CUSTOMER_ID'].str.split('//', expand=True))

    selected_columns = [
        'REPORT_DATE','CUSTOMER_ID','BRANCH','CUSTOMER_NO','CUSTOMER_EX_NO','PRODUCT_CODE',
        'CURRENCY','EFFECTIVE_DATE','MATURITY','NOTIONAL',
        'LAST_REPORT_DATE','MATURITY_CONT','MATURITY_NEW','NEXT_BUS_DATE',
        'Maturity_Periods_D','Maturity_Periods_M','SURV_TIME',
        'DIFF_TIME','DIFF_MONTH','TIME_BUCKET','Prepayment_Status'
    ]
    data = data[selected_columns]

    data = data[['REPORT_DATE','NOTIONAL','BRANCH','CURRENCY','PRODUCT_CODE','TIME_BUCKET','SURV_TIME','Prepayment_Status']].copy()
    data = data.rename(columns={
        'REPORT_DATE': 'Report_Date', 'NOTIONAL':'Notional',
        'BRANCH':'Branch','CURRENCY':'Currency','PRODUCT_CODE':'Product_Code',
        'TIME_BUCKET':'Time_Bucket','SURV_TIME':'Survival_in_Days','Prepayment_Status':'Status'
    })
    data['Status_TF'] = data['Status'].replace({'Early': True, 'OnTime': False})
    data['Status']    = data['Status'].replace({'Early': 1, 'OnTime': 0})
    data['Status_YN'] = data['Status'].replace({1:'Yes', 0:'No'})
    data['Prepayment_Notional'] = data['Notional'] * (data['Status'] == 1)
    return data

def filter_prepayment(df, branch=None, product=None, time_bucket=None, currency=None):
    data = df.copy()
    # Use same step-wise logic
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
    elif branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
    elif branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
    return data

# --------------------------- Early Withdrawal analysis (from early_withdrawal_analysis.py) ---------------------------

def data_model_early_withdrawal(data, monthly_bd_df, turkey_holidays_df, data_year_basis=None, start_date=None, end_date=None):
    data = data.copy()
    data.columns = data.columns.astype(str).str.replace(".", "_")
    data = data[~data["REPORT_DATE"].isin(pd.to_datetime([start_date, end_date])) & data["EFFECTIVE_DATE"].notna()]

    Max_Report_Date = (
        data.groupby('CUSTOMER_ID')['REPORT_DATE']
        .agg(MAX_RPT='max', N_RPT='size')
        .reset_index()
    )
    Max_Report_Date['CUSTOMER_IDs'] = Max_Report_Date['CUSTOMER_ID']
    Max_Report_Date = Max_Report_Date.set_index('CUSTOMER_IDs')
    Max_Report_Date.index.names = ['CUSTOMER_ID']

    data['REPORT_DATE'] = pd.to_datetime(data['REPORT_DATE'])

    if data_year_basis is not None:
        df_rolling_dates = rolling_dates(monthly_bd_df, years=data_year_basis)
        start_date = df_rolling_dates['START_DATE'].iloc[-1]
        end_date = df_rolling_dates['END_DATE'].iloc[-1]
        data = data[(data['REPORT_DATE'] >= start_date) & (data['REPORT_DATE'] <= end_date)].reset_index(drop=True)

    if start_date is not None and end_date is not None:
        data = data[(data['REPORT_DATE'] >= pd.to_datetime(start_date)) & (data['REPORT_DATE'] <= pd.to_datetime(end_date))].reset_index(drop=True)

    currency_series = data["CURRENCY"].copy()

    def _nx(d): return next_business_day(d, turkey_holidays_df)

    data = (
        data.assign(
            NO=lambda x: np.where(x['REPORT_DATE'] == x['CUSTOMER_ID'].map(Max_Report_Date.set_index('CUSTOMER_ID')['MAX_RPT']), 1, 0)
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
            MATURITY_CONT=lambda x: np.where(pd.to_datetime(x['MATURITY']).dt.weekday >= 5, 1, 0),
            MATURITY_NEW=lambda x: np.where(
                pd.to_datetime(x['MATURITY']).dt.weekday == 5, pd.to_datetime(x['MATURITY']) + pd.Timedelta(days=2),
                np.where(pd.to_datetime(x['MATURITY']).dt.weekday == 6, pd.to_datetime(x['MATURITY']) + pd.Timedelta(days=1), pd.to_datetime(x['MATURITY']))
            ),
            NEXT_BUS_DATE=lambda x: pd.to_datetime(x['REPORT_DATE']).apply(lambda d: _nx(d)),
            Maturity_Periods_D=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days,
            Maturity_Periods_M=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days / (365.25 / 12),
            SURV_TIME=lambda x: (pd.to_datetime(x['NEXT_BUS_DATE']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days,
            DIFF_TIME=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['NEXT_BUS_DATE'])).dt.days,
            DIFF_MONTH=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - (pd.to_datetime(x['MATURITY_NEW']) - pd.DateOffset(months=1))).dt.days,
            TIME_BUCKET=lambda x: x['Maturity_Periods_M'].apply(classify_time_bucket_core)
        )
        .assign(
            Early_Withdrawal_Status=lambda x: np.where(
                (pd.to_datetime(x['NEXT_BUS_DATE']) < pd.to_datetime(x['MATURITY_NEW'])) & (x['DIFF_TIME'] > x['DIFF_MONTH']),
                "Early", "OnTime"
            )
        )
    )

    data["CURRENCY"] = currency_series.loc[data.index].values
    data = data[data['DIFF_TIME'] >= 0]
    data[['BRANCH', 'CUSTOMER_NO','CUSTOMER_EX_NO', 'PRODUCT_CODE']] = (data['CUSTOMER_ID'].str.split('//', expand=True))

    selected_columns = ['REPORT_DATE','BRANCH','CURRENCY','PRODUCT_CODE','TIME_BUCKET','NOTIONAL','SURV_TIME','Early_Withdrawal_Status']
    data = data[selected_columns].copy()
    data = data.rename(columns={
        'REPORT_DATE':'Report_Date','BRANCH':'Branch','CURRENCY':'Currency','PRODUCT_CODE':'Product_Code',
        'TIME_BUCKET':'Time_Bucket','NOTIONAL':'Notional','SURV_TIME':'Survival_in_Days','Early_Withdrawal_Status':'Status'
    })
    data['Status_TF'] = data['Status'].replace({'Early': True, 'OnTime': False})
    data['Status'] = data['Status'].replace({'Early': 1, 'OnTime': 0})
    data['Early_Withdrawal_Notional'] = data['Notional'] * (data['Status'] == 1)
    return data

def filter_early_withdrawal(df, branch=None, product=None, time_bucket=None, currency=None):
    data = df.copy()
    # Step-wise logic
    if branch is not None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is None and product is not None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Product_Code'] == product) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is not None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is not None and product is not None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Product_Code'] == product) & (data['Currency'] == currency)]
    elif branch is None and product is None and time_bucket is not None and currency is not None:
        data = data.loc[(data['Time_Bucket'] == time_bucket) & (data['Currency'] == currency)]
    elif branch is not None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Branch'] == branch) & (data['Currency'] == currency)]
    elif branch is None and product is None and time_bucket is None and currency is not None:
        data = data.loc[(data['Currency'] == currency)]
    return data

# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="FRTB — Core/Prepayment/Early Withdrawal", layout="wide")
st.title("FRTB Analyses — Single App (Uploads only, no static paths)")

with st.sidebar:
    st.header("Choose Analysis")
    analysis = st.selectbox(
        "Analysis Type",
        ["Core Deposit", "Prepayment", "Early Withdrawal"]
    )

    st.markdown("---")
    st.subheader("User Inputs (# ENTERED BY USER)")
    start_date_str = st.text_input('start_date (ISO, e.g., "2017-01-01")', "")
    end_date_str   = st.text_input('end_date (ISO, e.g., "2025-12-31")', "")

    branch  = _none_if_empty(st.text_input("branch (string or empty)", ""))
    product = _none_if_empty(st.text_input("product (string or empty)", ""))
    time_bucket = _none_if_empty(st.text_input("time_bucket (string or empty; for Core it is integer-like)", ""))
    currency = _none_if_empty(st.text_input('currency (e.g., "TRY"; empty->None)', ""))

    if analysis == "Core Deposit":
        freq = st.selectbox("freq", ["W", "M", "Y", "D"], index=0)
        type_ = st.selectbox("type", ["last", "mean", "roll"], index=0)
    else:
        freq = None
        type_ = None

    # Convert
    start_date = _parse_date(start_date_str)
    end_date   = _parse_date(end_date_str)

st.markdown("### Upload Files")
hol_file = st.file_uploader("Turkey_Holidays.xlsx", type=["xlsx"], key="hol_file")

if analysis == "Core Deposit":
    data_file = st.file_uploader("CORE_DEPOSIT_DATA.xlsx", type=["xlsx"], key="core_file")
elif analysis == "Prepayment":
    data_file = st.file_uploader("PREPAYMENT_DATA.xlsx", type=["xlsx"], key="prep_file")
else:
    data_file = st.file_uploader("EARLY_WITHDRAWAL_DATA.xlsx", type=["xlsx"], key="ew_file")

run_btn = st.button("Run Analysis")

if run_btn:
    if hol_file is None or data_file is None:
        st.error("Please upload both the Turkey Holidays file and the corresponding data file for the selected analysis.")
        st.stop()

    # Read uploads
    try:
        turkey_holidays_df = pd.read_excel(hol_file)
    except Exception as e:
        st.exception(e)
        st.stop()

    try:
        data_df = pd.read_excel(data_file)
    except Exception as e:
        st.exception(e)
        st.stop()

    # Build calendars (shared)
    try:
        turkey_holidays_df, monthly_bd_df = build_business_cal(start_date, end_date, turkey_holidays_df)
    except Exception as e:
        st.error(f"Holiday calendar error: {e}")
        st.stop()

    # ---- CORE DEPOSIT ----
    if analysis == "Core Deposit":
        try:
            # Adapt exactly as scripts: data -> model -> clean/reshape -> filter -> analysis
            model_df = data_model_core(data_df, monthly_bd_df, data_year_basis=8, start_date=None, end_date=None)

            # Drop specific dates as in original script (if present)
            dd = model_df.copy()
            dd = dd[~(dd["REPORT_DATE"].astype(str).isin(["2020-12-08", "2023-12-27"]))]
            dd = reshape_core(dd)

            # time_bucket for core is numeric "0" in original—for compatibility, if empty -> None
            tb = None if time_bucket is None else time_bucket
            if tb is not None:
                try:
                    tb = int(tb)
                except Exception:
                    st.warning("time_bucket for Core should be integer-like (e.g., 0). Using None instead.")
                    tb = None

            filt = filter_core(dd, branch=branch, product=product, time_bucket=tb, currency=currency)

            results = core_deposit_analysis(filt, freq=freq or 'W', type_=type_ or 'last', nsim=100000)

            df_plot = results["df_resampled"]
            df_core = results["df_core"].flatten()
            core_pct = results["core_pct"]
            core_amount = results["core_amount"]
            min_month_year = results["min_month_year"]
            max_month_year = results["max_month_year"]
            mean_days, mean_months, mean_years = results["mean_days"], results["mean_months"], results["mean_years"]
            horizon_days, horizon_months, horizon_years = results["horizon_days"], results["horizon_months"], results["horizon_years"]

            st.success("Core Deposit analysis completed.")

            if go is None or make_subplots is None:
                st.warning("plotly is not available, skipping charts.")
            else:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Notional / Core Deposit", "Persistence"))
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Notional'], mode='lines', name='Notional / Core Deposit'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_core, mode='lines', fill='tozeroy', name='Core'), row=1, col=1)

                # Persistence panel (x in business-day-like units)
                pdays = results["persistence_days"]
                persistence = results["persistence_series"]
                fig.add_trace(go.Scatter(x=np.arange(len(persistence)), y=persistence, mode='lines', fill='tozeroy', name='Persistence'), row=1, col=2)
                fig.add_vline(x=pdays, line_width=2, line_dash="dot", line_color="royalblue", row=1, col=2)

                fig.update_layout(title=f"Core Deposit Analysis ({min_month_year} - {max_month_year})", width=1100, height=500, showlegend=False)
                fig.update_xaxes(title_text="Date", row=1, col=1)
                fig.update_yaxes(title_text="Demand Deposit", row=1, col=1)
                st.plotly_chart(fig, use_container_width=True)

            # Tables
            table_a = pd.DataFrame({
                "Core Deposit %": [round(core_pct*100, 2)],
                "Core Deposits (Mio)": [round(core_amount/1_000_000, 0)]
            })
            table_b = pd.DataFrame({
                "Mean Life (Days)": [mean_days],
                "Mean Life (Months)": [mean_months],
                "Mean Life (Years)": [mean_years]
            })
            table_c = pd.DataFrame({
                "Horizon Life (Days)": [horizon_days],
                "Horizon Life (Months)": [horizon_months],
                "Horizon Life (Years)": [horizon_years]
            })

            st.markdown("#### Results")
            st.dataframe(table_a, use_container_width=True)
            st.dataframe(table_b, use_container_width=True)
            st.dataframe(table_c, use_container_width=True)

        except Exception as e:
            st.exception(e)

    # ---- PREPAYMENT ----
    elif analysis == "Prepayment":
        try:
            data_df.columns = data_df.columns.astype(str).str.replace(".", "_")
            prem = data_model_prepayment(data_df, monthly_bd_df, turkey_holidays_df, data_year_basis=8, start_date=start_date, end_date=end_date)
            df_filtered = filter_prepayment(prem, branch=branch, product=product, time_bucket=time_bucket, currency=currency)

            # Kaplan-Meier
            t, s, ci, median_surv, median_surv_adj = kaplan_meier_estimator_custom(df_filtered)

            # Plot
            if go is None or make_subplots is None:
                st.warning("plotly is not available, skipping charts.")
            else:
                fig = make_subplots(rows=1, cols=1, subplot_titles=("",))
                fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name='Notional / Core Deposit'))
                y_min = round(float(np.min(s)), 2)
                fig.update_layout(title="Prepayment Analysis", width=800, height=500, showlegend=False, yaxis=dict(range=[y_min, 1]))
                fig.add_vline(x=median_surv_adj, line_width=2, line_dash="dot", line_color="royalblue")
                fig.add_vline(x=365.25, line_width=2, line_dash="dot", line_color="firebrick")
                # Add horizontal at 1-year probability
                one_year_prob = float(np.interp(365.25, t[::1], s[::1]))
                fig.add_hline(y=round(one_year_prob, 2), line_width=2, line_dash="dot", line_color="firebrick")
                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Probability of Survival")
                st.plotly_chart(fig, use_container_width=True)

            # Tables
            one_year_surv = round(float(np.interp(365.25, t[::1], s[::1])) * 100, 2)
            one_year_prepay = round((1 - float(np.interp(365.25, t[::1], s[::1]))) * 100, 2)

            table = pd.DataFrame({
                "1-year Survival Probability (95% CI)": [one_year_surv],
                "1-year Prepayment Probability (95% CI)": [one_year_prepay],
                "Survival Time (Adjusted)": [round(float(median_surv_adj), 0)],
                "Survival Time (Probability)": [round(float(np.interp(float(median_surv_adj), t[::1], s[::1])) * 100, 2)],
                "Survival Time (Max)": [round(float(median_surv), 0)]
            })
            st.markdown("#### Results")
            st.dataframe(table, use_container_width=True)
        except Exception as e:
            st.exception(e)

    # ---- EARLY WITHDRAWAL ----
    else:
        try:
            # As in original, there is an initial cleanup before modeling already inside the function
            ew = data_model_early_withdrawal(data_df, monthly_bd_df, turkey_holidays_df, data_year_basis=8, start_date=start_date, end_date=end_date)
            df_filtered = filter_early_withdrawal(ew, branch=branch, product=product, time_bucket=time_bucket, currency=currency)

            # Kaplan-Meier
            t, s, ci, median_surv, median_surv_adj = kaplan_meier_estimator_custom(df_filtered)

            # Plot
            if go is None or make_subplots is None:
                st.warning("plotly is not available, skipping charts.")
            else:
                fig = make_subplots(rows=1, cols=1, subplot_titles=("",))
                fig.add_trace(go.Scatter(x=t, y=s, mode='lines', name='Notional / Core Deposit'))
                y_min = round(float(np.min(s)), 2)
                fig.update_layout(title="Early Withdrawal Analysis", width=800, height=500, showlegend=False, yaxis=dict(range=[y_min, 1]))
                fig.add_vline(x=median_surv_adj, line_width=2, line_dash="dot", line_color="royalblue")
                fig.update_xaxes(title_text="Time")
                fig.update_yaxes(title_text="Probability of Survival")
                st.plotly_chart(fig, use_container_width=True)

            # Tables
            one_year_surv = round(float(np.interp(365.25, t[::1], s[::1])) * 100, 2)
            one_year_ew = round((1 - float(np.interp(365.25, t[::1], s[::1]))) * 100, 2)

            table = pd.DataFrame({
                "1-year Survival Probability (95% CI)": [one_year_surv],
                "1-year Early Withdrawal Probability (95% CI)": [one_year_ew],
                "Survival Time (Adjusted)": [round(float(median_surv_adj), 0)],
                "Survival Time (Probability)": [round(float(np.interp(float(median_surv_adj), t[::1], s[::1])) * 100, 2)],
                "Survival Time (Max)": [round(float(median_surv), 0)]
            })
            st.markdown("#### Results")
            st.dataframe(table, use_container_width=True)
        except Exception as e:
            st.exception(e)

else:
    st.info("Upload the two files and press **Run Analysis**.")

