"""
early_withdrawal_analysis.py
----------------------------
Notebook-parity Early Withdrawal Analysis, packaged for Streamlit.

Key features
- Accepts Excel uploads for data and Turkey holidays
- Runs the early-withdrawal pipeline (vectorized, robust to column names)
- Returns result DataFrames for display
- Optionally writes an .RData with key objects (via pyreadr) for consumption by R-only scripts

Usage (inside Streamlit):
    import early_withdrawal_analysis as ewa
    results = ewa.run(
        data_file=uploaded_data,          # path or BytesIO from st.file_uploader
        holidays_file=uploaded_holidays,  # path or BytesIO
        start_date=start_date,            # e.g. '2017-01-01' or None
        end_date=end_date,                # e.g. '2025-12-31' or None
        years=8,                          # rolling window in years (as in notebook)
        write_rdata=True,
        rdata_path="early_withdrawal_output.RData"
    )
    # results is a dict with keys:
    # 'report_df', 'analysis_df', 'ratio_df', 'monthly_bd_df', 'holidays_df', 'rdata_path'
"""

from __future__ import annotations

import io
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional dep for writing RData (no R install required)
try:
    import pyreadr  # type: ignore
except Exception:  # pragma: no cover
    pyreadr = None

warnings.simplefilter("ignore")

# ---------------------- Helpers ----------------------

def _read_excel_any(obj: Union[str, bytes, io.BytesIO], **kwargs) -> pd.DataFrame:
    """Read Excel from path or file-like; surface nicer errors."""
    try:
        if isinstance(obj, (bytes, bytearray)):
            obj = io.BytesIO(obj)
        return pd.read_excel(obj, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Excel okunamadı: {e}") from e

def _coerce_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(".", "_", regex=False)
    return df

def build_business_calendar(start_date: Optional[Union[str, datetime, pd.Timestamp]],
                            end_date: Optional[Union[str, datetime, pd.Timestamp]],
                            holidays_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (holidays_df_norm, monthly_bd_df)."""
    th = holidays_df.copy()
    th.columns = [str(c).strip() for c in th.columns]
    if 'TURKEY_HOLIDAYS' not in th.columns:
        # accept first column as holidays
        th = th.rename(columns={th.columns[0]: 'TURKEY_HOLIDAYS'})
    th['TURKEY_HOLIDAYS'] = pd.to_datetime(th['TURKEY_HOLIDAYS']).dt.date

    # Defaults if None
    if start_date is None:
        start_date = datetime(2010, 1, 1)
    if end_date is None:
        end_date = datetime(2030, 12, 31)
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    bd = pd.bdate_range(start=start_date, end=end_date).date
    # Exclude holidays
    holiday_set = set(th['TURKEY_HOLIDAYS'])
    bd = pd.to_datetime([d for d in bd if d not in holiday_set])

    monthly_bd = (
        pd.DataFrame({"Business_Days": bd})
        .assign(YearMonth=lambda df: df["Business_Days"].dt.to_period("M"))
        .groupby("YearMonth")["Business_Days"].agg(["first", "last"]).reset_index()
    )
    return th, monthly_bd

def next_business_day(date_val, holidays_df: pd.DataFrame):
    if pd.isna(date_val):
        return date_val
    if holidays_df.shape[1] == 0:
        holidays = set()
    else:
        holidays = set(pd.to_datetime(holidays_df.iloc[:, 0]).dt.date)
    d = pd.to_datetime(date_val).date()
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5 or nxt in holidays:
        nxt += timedelta(days=1)
    return nxt

def classify_time_bucket(months: float) -> str:
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

def rolling_dates(monthly_bd_df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    period = years * 12
    rows = []
    for i in range(len(monthly_bd_df) - (period - 1)):
        start_d = monthly_bd_df.loc[i, "first"]
        end_d = monthly_bd_df.loc[i + (period - 1), "last"]
        rows.append({"START_DATE": start_d, "END_DATE": end_d})
    return pd.DataFrame(rows)

# ---------------------- Core pipeline ----------------------

def _data_model(df: pd.DataFrame,
                monthly_bd_df: pd.DataFrame,
                holidays_df: pd.DataFrame,
                data_year_basis: Optional[int] = None,
                start_date: Optional[Union[str, datetime]] = None,
                end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
    """Replicates notebook logic to a tidy frame for early-withdrawal analysis."""
    df = _coerce_colnames(df)
    # enforce required columns with friendly error
    required = {
        'REPORT_DATE', 'CUSTOMER_ID', 'EFFECTIVE_DATE', 'MATURITY', 'NOTIONAL', 'CURRENCY'
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}. Bu kolonlar zorunlu: {sorted(required)}")

    df['REPORT_DATE']   = pd.to_datetime(df['REPORT_DATE'])
    df['EFFECTIVE_DATE'] = pd.to_datetime(df['EFFECTIVE_DATE'])
    df['MATURITY']       = pd.to_datetime(df['MATURITY'])

    # OPTIONAL: restrict to rolling window
    if data_year_basis is not None:
        rd = rolling_dates(monthly_bd_df, years=data_year_basis)
        start_date = rd['START_DATE'].iloc[-1]
        end_date   = rd['END_DATE'].iloc[-1]

    if start_date is not None and end_date is not None:
        df = df[(df['REPORT_DATE'] >= pd.to_datetime(start_date)) & (df['REPORT_DATE'] <= pd.to_datetime(end_date))].reset_index(drop=True)

    # Keep last report per customer_id
    max_rpt = (
        df.groupby('CUSTOMER_ID')['REPORT_DATE']
          .agg(MAX_RPT='max', N_RPT='size')
          .reset_index()
    )
    # shortcut map
    mx = max_rpt.set_index('CUSTOMER_ID')['MAX_RPT']

    # compute features on the last record per customer
    def _nx(d): return next_business_day(d, holidays_df)

    df2 = (
        df.assign(NO=lambda x: np.where(x['REPORT_DATE'] == x['CUSTOMER_ID'].map(mx), 1, 0))
          .query('NO == 1')
          .assign(LAST_REPORT_DATE=lambda x: x['REPORT_DATE'].max())
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
                  np.where(pd.to_datetime(x['MATURITY']).dt.weekday == 6, pd.to_datetime(x['MATURITY']) + pd.Timedelta(days=1),
                           pd.to_datetime(x['MATURITY']))
              ),
              NEXT_BUS_DATE=lambda x: pd.to_datetime(x['REPORT_DATE']).apply(_nx),
              Maturity_Periods_D=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days,
              Maturity_Periods_M=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days / (365.25/12),
              SURV_TIME=lambda x: (pd.to_datetime(x['NEXT_BUS_DATE']) - pd.to_datetime(x['EFFECTIVE_DATE'])).dt.days,
              DIFF_TIME=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - pd.to_datetime(x['NEXT_BUS_DATE'])).dt.days,
              DIFF_MONTH=lambda x: (pd.to_datetime(x['MATURITY_NEW']) - (pd.to_datetime(x['MATURITY_NEW']) - pd.DateOffset(months=1))).dt.days,
              TIME_BUCKET=lambda x: x['Maturity_Periods_M'].apply(classify_time_bucket)
          )
          .assign(
              Early_Withdrawal_Status=lambda x: np.where(
                  (pd.to_datetime(x['NEXT_BUS_DATE']) < pd.to_datetime(x['MATURITY_NEW'])) & (x['DIFF_TIME'] > x['DIFF_MONTH']),
                  'Early', 'OnTime'
              )
          )
    )

    # Recover currency by customer from original df (last seen currency per id)
    cur_map = df.sort_values('REPORT_DATE').groupby('CUSTOMER_ID')['CURRENCY'].last()
    df2['CURRENCY'] = df2['CUSTOMER_ID'].map(cur_map)

    # Clean negatives
    df2 = df2[df2['DIFF_TIME'] >= 0].copy()

    # Split CUSTOMER_ID if format 'BRANCH//CUSTOMER_NO//...//PRODUCT_CODE'
    parts = df2['CUSTOMER_ID'].astype(str).str.split('//', expand=True)
    for i, name in enumerate(['BRANCH','CUSTOMER_NO','CUSTOMER_EX_NO','PRODUCT_CODE']):
        if i < parts.shape[1]:
            df2[name] = parts[i].str.strip()
        else:
            df2[name] = pd.NA

    # Final tidy table
    out = (df2[['REPORT_DATE','BRANCH','CURRENCY','PRODUCT_CODE','TIME_BUCKET','NOTIONAL','SURV_TIME','Early_Withdrawal_Status']]
              .rename(columns={
                  'REPORT_DATE':'Report_Date','CURRENCY':'Currency','PRODUCT_CODE':'Product_Code',
                  'TIME_BUCKET':'Time_Bucket','NOTIONAL':'Notional','SURV_TIME':'Survival_in_Days',
                  'Early_Withdrawal_Status':'Status'
              })
           )
    out['Status_TF'] = out['Status'].map({'Early': True, 'OnTime': False})
    out['Status']    = out['Status'].map({'Early': 1, 'OnTime': 0})
    out['Early_Withdrawal_Notional'] = out['Notional'] * (out['Status'] == 1)
    return out

# ---------------------- Public API ----------------------

def run(data_file: Union[str, bytes, io.BytesIO],
        holidays_file: Union[str, bytes, io.BytesIO],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        years: int = 8,
        write_rdata: bool = True,
        rdata_path: str = "early_withdrawal_output.RData") -> Dict[str, Any]:
    """Run Early Withdrawal Analysis end-to-end.

    Returns dict with:
      - report_df: tidy record-level table used for aggregations
      - analysis_df: daily analysis by Report_Date x Branch x Currency x Product_Code x Time_Bucket
      - ratio_df: overall ratios by Branch x Currency x Product_Code x Time_Bucket
      - monthly_bd_df, holidays_df
      - rdata_path (if write_rdata)
    """
    # Read inputs
    raw_df = _read_excel_any(data_file)
    hol_df = _read_excel_any(holidays_file)
    hol_df, monthly_bd = build_business_calendar(start_date, end_date, hol_df)

    # Build the tidy record table
    report_df = _data_model(raw_df, monthly_bd, hol_df, data_year_basis=years,
                            start_date=start_date, end_date=end_date)

    # Analysis frames
    analysis_df = (
        report_df.groupby(['Report_Date','Branch','Currency','Product_Code','Time_Bucket'], as_index=False)
                 .agg(Notional=("Notional", "sum"),
                      Early_Withdrawal_Notional=("Early_Withdrawal_Notional", "sum"),
                      N_Count=("Product_Code", "size"),
                      Early_Withdrawal_Cust_Ratio=('Status', lambda x: np.round((x == 1).sum() / len(x) * 100, 2)))
    )
    analysis_df['Early_Withdrawal_Not_Ratio'] = np.round(
        np.where(analysis_df['Notional'] == 0, np.nan,
                 analysis_df['Early_Withdrawal_Notional'] / analysis_df['Notional'] * 100), 2
    )

    ratio_df = (
        report_df.groupby(['Branch','Currency','Product_Code','Time_Bucket'])['Early_Withdrawal_Notional'].sum() \
                 .div(report_df.groupby(['Branch','Currency','Product_Code','Time_Bucket'])['Notional'].sum()) \
                 .reset_index(name='Early_Withdrawal_Not_Ratio') \
                 .round(4)
    )

    # Optionally write an .RData
    rdata_out = None
    if write_rdata:
        if pyreadr is None:
            raise ImportError("pyreadr kurulu değil. 'pip install pyreadr' ile kurup tekrar deneyin.")
        # Save selected objects so the R-only script can read the same names
        objects = {
            'Report_Early_Withdrawal': report_df,
            'Early_Withdrawal_Analysis': analysis_df,
            'Early_Withdrawal_Notional_Ratio': ratio_df,
            'Monthly_Business_Days': monthly_bd,
            'Turkey_Holidays': pd.DataFrame({'TURKEY_HOLIDAYS': pd.to_datetime(hol_df['TURKEY_HOLIDAYS'])})
        }
        pyreadr.write_rdata(rdata_path, objects)  # writes a .RData with multiple objects
        rdata_out = rdata_path

    return {
        'report_df': report_df,
        'analysis_df': analysis_df,
        'ratio_df': ratio_df,
        'monthly_bd_df': monthly_bd,
        'holidays_df': hol_df,
        'rdata_path': rdata_out,
    }
